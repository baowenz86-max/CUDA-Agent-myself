#include <cuda_runtime.h>
#include <math.h>

// Batchnorm inference kernel
__global__ void batchnorm_inference_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N,
    int C,
    int HW
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < (int64_t)N * C * HW; i += stride) {
        int n = i / (C * HW);
        int c = (i / HW) % C;
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float gamma = (weight != nullptr) ? weight[c] : 1.0f;
        float beta = (bias != nullptr) ? bias[c] : 0.0f;
        float val = (x[i] - mean) * inv_std * gamma + beta;
        out[i] = val;
    }
}

// Triplet margin loss kernel: one block per item, each thread accumulates over a strided chunk
// Shared memory reduction with proper initialization to zero for all threads
__global__ void triplet_margin_loss_sum_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ out,
    float margin,
    int N,
    int D
) {
    int item = blockIdx.x;
    if (item >= N) return;

    const float* a = anchor + (int64_t)item * D;
    const float* p = positive + (int64_t)item * D;
    const float* n = negative + (int64_t)item * D;

    float sum_pos = 0.0f;
    float sum_neg = 0.0f;

    // Each thread accumulates over a strided chunk
    for (int idx = threadIdx.x; idx < D; idx += blockDim.x) {
        float diff_pos = a[idx] - p[idx];
        float diff_neg = a[idx] - n[idx];
        sum_pos += diff_pos * diff_pos;
        sum_neg += diff_neg * diff_neg;
    }

    // Shared memory for reduction, initialize all entries to zero
    __shared__ float s_pos[256];
    __shared__ float s_neg[256];
    s_pos[threadIdx.x] = 0.0f;
    s_neg[threadIdx.x] = 0.0f;
    __syncthreads();

    // Write local sums into shared memory
    s_pos[threadIdx.x] = sum_pos;
    s_neg[threadIdx.x] = sum_neg;
    __syncthreads();

    // Tree reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_pos[threadIdx.x] += s_pos[threadIdx.x + s];
            s_neg[threadIdx.x] += s_neg[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float dist_pos = sqrtf(s_pos[0]);
        float dist_neg = sqrtf(s_neg[0]);
        float loss_i = fmaxf(dist_pos - dist_neg + margin, 0.0f);
        atomicAdd(out, loss_i);
    }
}

// Launcher for batchnorm
void batchnorm_inference_launcher(
    const float* x,
    float* out,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N,
    int C,
    int HW,
    cudaStream_t stream
) {
    int64_t total = (int64_t)N * C * HW;
    int block_size = 256;
    int grid_size = (int)min((total + block_size - 1) / block_size, (int64_t)65535);
    if (grid_size < 1) grid_size = 1;
    batchnorm_inference_kernel<<<grid_size, block_size, 0, stream>>>(
        x, out, weight, bias, running_mean, running_var, eps, N, C, HW);
}

// Launcher for triplet margin loss
void triplet_margin_loss_sum_launcher(
    const float* anchor,
    const float* positive,
    const float* negative,
    float* out,
    float margin,
    int N,
    int D,
    cudaStream_t stream
) {
    // Zero output before accumulation
    cudaMemsetAsync(out, 0, sizeof(float), stream);

    constexpr int BLOCK_DIM = 256;
    triplet_margin_loss_sum_kernel<<<N, BLOCK_DIM, 0, stream>>>(
        anchor, positive, negative, out, margin, N, D);
}
