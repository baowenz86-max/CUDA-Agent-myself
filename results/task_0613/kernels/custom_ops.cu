#include <cuda_runtime.h>

// CUDA kernel for InstanceNorm1d over last dimension.
// x: input tensor, shape [N, C, L], may be non-contiguous.
// weight, bias: per-channel affine parameters, shape [C], contiguous.
// out: output tensor, same shape as x, contiguous.
// The kernel handles one (n,c) pair per thread block, computes mean and variance over L using one block, then normalizes.

template <int BLOCK_SIZE>
__global__ void instance_norm1d_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int N, int C, int L,
    long long x_stride_n, long long x_stride_c, long long x_stride_l,
    long long out_stride_n, long long out_stride_c, long long out_stride_l)
{
    // Block index corresponds to (n, c) pair, but may need grid-stride loop if N*C > gridDim.x
    int total_pairs = N * C;
    for (int pair_idx = blockIdx.x; pair_idx < total_pairs; pair_idx += gridDim.x) {
        int n = pair_idx / C;
        int c = pair_idx % C;

        // Base offsets for this pair
        const float* x_base = x + n * x_stride_n + c * x_stride_c;
        float* out_base = out + n * out_stride_n + c * out_stride_c;

        // Compute mean via shared memory reduction
        __shared__ float s_sum[BLOCK_SIZE];
        __shared__ float s_mean;
        __shared__ float s_var;

        float local_sum = 0.0f;
        // Each thread handles a strided subset of L
        for (int l = threadIdx.x; l < L; l += BLOCK_SIZE) {
            local_sum += x_base[l * x_stride_l];
        }

        // Block reduction for sum
        s_sum[threadIdx.x] = local_sum;
        __syncthreads();

        for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                s_sum[threadIdx.x] += s_sum[threadIdx.x + offset];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            s_mean = s_sum[0] / (float)L;
        }
        __syncthreads();

        float mean = s_mean;

        // Compute variance (biased) via shared memory reduction of squared differences
        float local_var_sum = 0.0f;
        for (int l = threadIdx.x; l < L; l += BLOCK_SIZE) {
            float diff = x_base[l * x_stride_l] - mean;
            local_var_sum += diff * diff;
        }

        s_sum[threadIdx.x] = local_var_sum;
        __syncthreads();

        for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                s_sum[threadIdx.x] += s_sum[threadIdx.x + offset];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            s_var = s_sum[0] / (float)L;
        }
        __syncthreads();

        float var = s_var;
        float inv_std = rsqrtf(var + eps);
        float w = weight[c];
        float b = bias[c];

        // Normalize and write output
        for (int l = threadIdx.x; l < L; l += BLOCK_SIZE) {
            float val = (x_base[l * x_stride_l] - mean) * inv_std;
            out_base[l * out_stride_l] = val * w + b;
        }
    }
}

extern "C" void instance_norm_forward_launcher(
    float* out,
    const float* x,
    const float* weight,
    const float* bias,
    float eps,
    int N, int C, int L,
    long long x_stride_n, long long x_stride_c, long long x_stride_l,
    long long out_stride_n, long long out_stride_c, long long out_stride_l,
    cudaStream_t stream)
{
    const int BLOCK_SIZE = 256;
    int total_pairs = N * C;
    // Use enough blocks to cover all pairs, but cap to avoid excessive blocks; kernel loops if needed.
    int blocks = (total_pairs + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > 65535) blocks = 65535; // grid limit on most GPUs, kernel will grid-stride loop
    instance_norm1d_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
        out, x, weight, bias, eps,
        N, C, L,
        x_stride_n, x_stride_c, x_stride_l,
        out_stride_n, out_stride_c, out_stride_l);
}
