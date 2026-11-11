#include <cuda_runtime.h>
#include <cstdint>

// embedding_lookup kernel: 对每个 (batch, seq) 索引，从 weight 中复制 embedding_dim 个 float
// 输入: input int64_t 指针，形状 batch*seq; weight float*，形状 num_embeddings*embedding_dim
// 输出: output float*，形状 batch*seq*embedding_dim
__global__ void embedding_lookup_kernel(const int64_t* __restrict__ input,
                                        const float* __restrict__ weight,
                                        float* __restrict__ output,
                                        int64_t total_tokens,       // batch*seq
                                        int64_t embedding_dim,
                                        int64_t num_embeddings) {
    // 每个线程处理一个输出元素，使用 grid-stride 循环覆盖所有 total_tokens*embedding_dim 个元素
    int64_t total_elements = total_tokens * embedding_dim;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t token_idx = i / embedding_dim;
        int64_t dim = i % embedding_dim;
        int64_t input_idx = input[token_idx];
        // 边界检查（虽然契约保证合法，但保持安全）
        if (input_idx >= 0 && input_idx < num_embeddings) {
            output[i] = weight[input_idx * embedding_dim + dim];
        } else {
            output[i] = 0.0f;
        }
    }
}

// l2_normalize kernel: 对每个 token 的 embedding 向量 (长度 embedding_dim) 执行 L2 归一化
// 输入 x: float* 形状 total_tokens*embedding_dim
// 输出 y: float* 相同形状
// epsilon: 加法项
// 采用每个 block 处理一个或少量 token，使用共享内存保存平方和，通过 reduction 计算范数
__global__ void l2_normalize_kernel(const float* __restrict__ x,
                                    float* __restrict__ y,
                                    int64_t total_tokens,
                                    int64_t embedding_dim,
                                    float epsilon) {
    // 每个 block 处理一个 token，shared memory 用于 reduction
    extern __shared__ float s_sum[]; // 大小 blockDim.x 个 float

    int64_t token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    const float* x_token = x + token_idx * embedding_dim;
    float* y_token = y + token_idx * embedding_dim;

    // 每个线程计算部分平方和
    float local_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
        float val = x_token[i];
        local_sum += val * val;
    }

    // 写入共享内存，然后 block 内归约
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // 归约（blockDim.x 为 2 的幂）
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float sum = s_sum[0];
    float inv_norm = rsqrtf(sum + epsilon);

    // 每个线程缩放对应元素
    for (int64_t i = threadIdx.x; i < embedding_dim; i += blockDim.x) {
        y_token[i] = x_token[i] * inv_norm;
    }
}

// Host launcher: embedding_lookup
// 参数: input (int64_t*), weight (float*), output (float*), total_tokens, embedding_dim, num_embeddings, stream
extern "C" void embedding_lookup_forward(const int64_t* input,
                                         const float* weight,
                                         float* output,
                                         int64_t total_tokens,
                                         int64_t embedding_dim,
                                         int64_t num_embeddings,
                                         cudaStream_t stream) {
    int64_t total_elements = total_tokens * embedding_dim;
    int threads = 256;
    int64_t blocks = (total_elements + threads - 1) / threads;
    // 限制 grid 大小，使用 grid-stride 循环
    if (blocks > 65535) blocks = 65535;
    embedding_lookup_kernel<<<(int)blocks, threads, 0, stream>>>(
        input, weight, output, total_tokens, embedding_dim, num_embeddings);
}

// Host launcher: l2_normalize
// 参数: x (float*), y (float*), total_tokens, embedding_dim, epsilon, stream
extern "C" void l2_normalize_forward(const float* x,
                                     float* y,
                                     int64_t total_tokens,
                                     int64_t embedding_dim,
                                     float epsilon,
                                     cudaStream_t stream) {
    // 每个 block 处理一个 token，block 大小取 256 或 embedding_dim 的较小值
    int threads = 256;
    if (embedding_dim < threads) threads = (int)embedding_dim;
    // 共享内存大小
    int shared_mem_size = threads * sizeof(float);
    int blocks = (int)total_tokens; // total_tokens 不会超过 int
    l2_normalize_kernel<<<blocks, threads, shared_mem_size, stream>>>(
        x, y, total_tokens, embedding_dim, epsilon);
}
