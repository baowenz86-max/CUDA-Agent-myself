#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

#define CHECK_CUDA_ERR()                                                                          \
    do {                                                                                           \
        cudaError_t err = cudaGetLastError();                                                      \
        if (err != cudaSuccess) {                                                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return;                                                                                 \
        }                                                                                          \
    } while (0)

// -----------------------------------------------------------------------------
// one_hot_forward
// -----------------------------------------------------------------------------
__global__ void one_hot_kernel(const int64_t* __restrict__ x, float* __restrict__ out,
                               int64_t batch_size, int64_t num_classes) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < batch_size; i += stride) {
        int64_t cls = x[i];
        float* row = out + i * num_classes;
        for (int64_t j = 0; j < num_classes; ++j) {
            row[j] = 0.0f;
        }
        if (cls >= 0 && cls < num_classes) {
            row[cls] = 1.0f;
        }
    }
}

// -----------------------------------------------------------------------------
// 辅助 kernel: 计算 log_softmax 沿最后一维，并将结果写入 out，size 为行数，dim 为列数
// -----------------------------------------------------------------------------
__global__ void log_softmax_kernel(const float* __restrict__ logits, float* __restrict__ out,
                                   int64_t rows, int64_t cols) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* src = logits + row * cols;
    float* dst = out + row * cols;

    // find max for numerical stability
    float max_val = -INFINITY;
    for (int64_t c = 0; c < cols; ++c) {
        float v = src[c];
        if (v > max_val) max_val = v;
    }

    float sum = 0.0f;
    for (int64_t c = 0; c < cols; ++c) {
        float e = expf(src[c] - max_val);
        dst[c] = e;
        sum += e;
    }
    float log_sum = logf(sum);
    for (int64_t c = 0; c < cols; ++c) {
        dst[c] = src[c] - max_val - log_sum;
    }
}

// -----------------------------------------------------------------------------
// 组合 kernel：将短头 log_prob 和 tail log_prob 合并到最终输出
// 输出矩阵 out: [batch_size, n_classes]
// head_logprob: [batch_size, head_num_classes]
// tail0_logprob: [batch_size, 500]
// tail1_logprob: [batch_size, 4200]
// cutoffs[0]=300, cutoffs[1]=800, head_num_classes=302
// -----------------------------------------------------------------------------
__global__ void combine_log_prob_kernel(const float* __restrict__ head_logprob,
                                        const float* __restrict__ tail0_logprob,
                                        const float* __restrict__ tail1_logprob,
                                        float* __restrict__ out,
                                        int64_t batch_size, int64_t n_classes,
                                        int64_t head_num_classes,
                                        int64_t cutoff0, int64_t cutoff1) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size) return;

    const float* head_row = head_logprob + row * head_num_classes;
    const float* tail0_row = tail0_logprob + row * 500;
    const float* tail1_row = tail1_logprob + row * 4200;
    float* out_row = out + row * n_classes;

    // 短头类别 0..299
    for (int64_t c = 0; c < cutoff0; ++c) {
        out_row[c] = head_row[c];
    }
    // 长尾簇 0：总类别 300..799，对应 head 索引 300 和 tail0 索引 0..499
    int64_t head_cluster0 = cutoff0; // 300
    for (int64_t c = 0; c < 500; ++c) {
        out_row[cutoff0 + c] = head_row[head_cluster0] + tail0_row[c];
    }
    // 长尾簇 1：总类别 800..4999，对应 head 索引 301 和 tail1 索引 0..4199
    int64_t head_cluster1 = cutoff0 + 1; // 301
    for (int64_t c = 0; c < 4200; ++c) {
        out_row[cutoff1 + c] = head_row[head_cluster1] + tail1_row[c];
    }
}

// -----------------------------------------------------------------------------
// 提取索引并计算 head_logits 和 hidden 的 kernel
// 输入 one_hot: [batch_size, in_features] float
// 输出 head_logits: [batch_size, head_num_classes]
// 输出 hidden0: [batch_size, 256]
// 输出 hidden1: [batch_size, 64]
// 权重：head_weight [head_num_classes, in_features], head_bias [head_num_classes]
// tail0_0_weight [256, in_features], tail1_0_weight [64, in_features]
// -----------------------------------------------------------------------------
__global__ void extract_and_first_layer_kernel(
    const float* __restrict__ x_one_hot,
    const float* __restrict__ head_weight,
    const float* __restrict__ head_bias,
    const float* __restrict__ tail0_0_weight,
    const float* __restrict__ tail1_0_weight,
    float* __restrict__ head_logits,
    float* __restrict__ hidden0,
    float* __restrict__ hidden1,
    int64_t batch_size, int64_t in_features,
    int64_t head_num_classes) {
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch_size) return;
    const float* one_hot_row = x_one_hot + row * in_features;
    // 找到非零索引
    int64_t idx = -1;
    for (int64_t k = 0; k < in_features; ++k) {
        if (one_hot_row[k] != 0.0f) {
            idx = k;
            break;
        }
    }
    // 如果没有找到（理论上不可能，但安全处理），设为 0
    if (idx == -1) idx = 0;

    // head_logits
    for (int64_t c = 0; c < head_num_classes; ++c) {
        head_logits[row * head_num_classes + c] = head_weight[c * in_features + idx] + head_bias[c];
    }
    // hidden0
    for (int64_t c = 0; c < 256; ++c) {
        hidden0[row * 256 + c] = tail0_0_weight[c * in_features + idx];
    }
    // hidden1
    for (int64_t c = 0; c < 64; ++c) {
        hidden1[row * 64 + c] = tail1_0_weight[c * in_features + idx];
    }
}

// -----------------------------------------------------------------------------
// 简单 matmul_nt kernel: C = A * B^T, A: MxK, B: NxK, C: MxN
// 使用共享内存分块，BLOCK_SIZE 可调
// -----------------------------------------------------------------------------
#define BLOCK_SIZE 16
__global__ void matmul_nt_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                 float* __restrict__ C,
                                 int64_t M, int64_t N, int64_t K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;
    for (int64_t kk = 0; kk < K; kk += BLOCK_SIZE) {
        // 加载 A tile
        if (row < M && kk + tx < K) {
            As[ty][tx] = A[row * K + kk + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        // 加载 B tile，注意 B 的元素是 B[col][kk+ty]
        if (col < N && kk + ty < K) {
            Bs[ty][tx] = B[col * K + kk + ty];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();
        // 计算部分和
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            sum += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// -----------------------------------------------------------------------------
// Launchers
// -----------------------------------------------------------------------------
extern "C" void one_hot_forward_launcher(const int64_t* x, float* out,
                                          int64_t batch_size, int64_t num_classes,
                                          cudaStream_t stream) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    one_hot_kernel<<<blocks, threads, 0, stream>>>(x, out, batch_size, num_classes);
    CHECK_CUDA_ERR();
}

extern "C" void adaptive_log_softmax_log_prob_forward_launcher(
    const float* x_one_hot,
    const float* head_weight,
    const float* head_bias,
    const float* tail0_0_weight,
    const float* tail0_1_weight,
    const float* tail1_0_weight,
    const float* tail1_1_weight,
    float* out,
    int64_t batch_size,
    int64_t in_features,
    int64_t head_num_classes,
    int64_t n_classes,
    int64_t cutoff0,
    int64_t cutoff1,
    float div_value,
    cudaStream_t stream) {
    (void)div_value; // 未使用

    // 分配临时设备内存
    float *head_logits, *head_logprob, *hidden0, *hidden1, *tail0_logits, *tail0_logprob, *tail1_logits, *tail1_logprob;
    cudaMalloc(&head_logits, batch_size * head_num_classes * sizeof(float));
    cudaMalloc(&head_logprob, batch_size * head_num_classes * sizeof(float));
    cudaMalloc(&hidden0, batch_size * 256 * sizeof(float));
    cudaMalloc(&hidden1, batch_size * 64 * sizeof(float));
    cudaMalloc(&tail0_logits, batch_size * 500 * sizeof(float));
    cudaMalloc(&tail0_logprob, batch_size * 500 * sizeof(float));
    cudaMalloc(&tail1_logits, batch_size * 4200 * sizeof(float));
    cudaMalloc(&tail1_logprob, batch_size * 4200 * sizeof(float));

    // 第一步：提取索引并计算 head_logits 和 hidden
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    extract_and_first_layer_kernel<<<blocks, threads, 0, stream>>>(
        x_one_hot, head_weight, head_bias, tail0_0_weight, tail1_0_weight,
        head_logits, hidden0, hidden1,
        batch_size, in_features, head_num_classes);
    CHECK_CUDA_ERR();

    // 计算 head log_softmax
    log_softmax_kernel<<<blocks, threads, 0, stream>>>(head_logits, head_logprob, batch_size, head_num_classes);
    CHECK_CUDA_ERR();

    // 计算 tail0 第二层 GEMM: tail0_logits = hidden0 * tail0_1_weight^T
    // hidden0: batch x 256, tail0_1_weight: 500 x 256
    dim3 blockDim(16, 16);
    dim3 gridDim((500 + 16 - 1) / 16, (batch_size + 16 - 1) / 16);
    matmul_nt_kernel<<<gridDim, blockDim, 0, stream>>>(hidden0, tail0_1_weight, tail0_logits, batch_size, 500, 256);
    CHECK_CUDA_ERR();

    // 计算 tail1 第二层 GEMM: tail1_logits = hidden1 * tail1_1_weight^T
    // hidden1: batch x 64, tail1_1_weight: 4200 x 64
    gridDim = dim3((4200 + 16 - 1) / 16, (batch_size + 16 - 1) / 16);
    matmul_nt_kernel<<<gridDim, blockDim, 0, stream>>>(hidden1, tail1_1_weight, tail1_logits, batch_size, 4200, 64);
    CHECK_CUDA_ERR();

    // 计算 tail0 log_softmax
    log_softmax_kernel<<<blocks, threads, 0, stream>>>(tail0_logits, tail0_logprob, batch_size, 500);
    CHECK_CUDA_ERR();

    // 计算 tail1 log_softmax
    log_softmax_kernel<<<blocks, threads, 0, stream>>>(tail1_logits, tail1_logprob, batch_size, 4200);
    CHECK_CUDA_ERR();

    // 组合输出
    combine_log_prob_kernel<<<blocks, threads, 0, stream>>>(
        head_logprob, tail0_logprob, tail1_logprob, out,
        batch_size, n_classes, head_num_classes, cutoff0, cutoff1);
    CHECK_CUDA_ERR();

    // 释放临时内存
    cudaFree(head_logits);
    cudaFree(head_logprob);
    cudaFree(hidden0);
    cudaFree(hidden1);
    cudaFree(tail0_logits);
    cudaFree(tail0_logprob);
    cudaFree(tail1_logits);
    cudaFree(tail1_logprob);
}
