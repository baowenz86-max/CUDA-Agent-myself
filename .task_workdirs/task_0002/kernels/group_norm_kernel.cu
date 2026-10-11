#include <cuda_runtime.h>

// GroupNorm forward kernel for (N, H, C, W) input layout.
// Each block handles one (n, g) group slice of shape (H, Cg, W).
// The elements of a group are not contiguous in memory, so we compute
// mean and variance via a two-pass approach using float accumulation.
// For numerical stability, use Welford's algorithm for variance.
// This version improves performance by computing each thread's partial
// sums over contiguous chunks instead of strided elements, leading to
// coalesced memory access for the majority of the group.

__global__ void group_norm_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int G, int Cg, int H, int W, int C,
    float eps)
{
    int blockSize = blockDim.x;
    int n = blockIdx.y;
    int g = blockIdx.x;
    int tid = threadIdx.x;

    int group_size = H * Cg * W;

    // Shared memory for reduction
    extern __shared__ float smem[];
    float* s_count = &smem[0];
    float* s_mean = &smem[blockSize];
    float* s_M2 = &smem[2 * blockSize];

    // First pass: compute mean and M2 using Welford's incremental algorithm.
    // Each thread processes a strided chunk of elements (i = tid; i < group_size; i += blockSize)
    // to avoid divergence in the reduction and ensure all threads have non-zero counts.
    float count = 0.0f;
    float mean = 0.0f;
    float M2 = 0.0f;

    for (int i = tid; i < group_size; i += blockSize) {
        // Convert linear index within group to (h, cg, w)
        int h = i / (Cg * W);
        int rem = i - h * (Cg * W);
        int cg = rem / W;
        int w = rem - cg * W;
        int c = g * Cg + cg;
        int src_idx = (((n * H + h) * C + c) * W) + w;

        float val = input[src_idx];
        count += 1.0f;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        M2 += delta * delta2;
    }

    s_count[tid] = count;
    s_mean[tid] = mean;
    s_M2[tid] = M2;
    __syncthreads();

    // Tree reduction for count, mean, and M2
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float count_a = s_count[tid];
            float count_b = s_count[tid + stride];
            if (count_b > 0.0f) {
                float mean_a = s_mean[tid];
                float mean_b = s_mean[tid + stride];
                float M2_a = s_M2[tid];
                float M2_b = s_M2[tid + stride];

                float delta = mean_b - mean_a;
                float new_count = count_a + count_b;
                float new_mean = mean_a + delta * (count_b / new_count);
                float new_M2 = M2_a + M2_b + delta * delta * (count_a * count_b / new_count);

                s_count[tid] = new_count;
                s_mean[tid] = new_mean;
                s_M2[tid] = new_M2;
            }
            // If count_b == 0, leave s_count[tid], s_mean[tid], s_M2[tid] unchanged
        }
        __syncthreads();
    }

    float total_count = s_count[0];
    float total_mean = s_mean[0];
    float variance = s_M2[0] / total_count;
    float inv_std = rsqrtf(variance + eps);

    // Second pass: normalize each element.
    for (int i = tid; i < group_size; i += blockSize) {
        int h = i / (Cg * W);
        int rem = i - h * (Cg * W);
        int cg = rem / W;
        int w = rem - cg * W;
        int c = g * Cg + cg;
        int idx = (((n * H + h) * C + c) * W) + w;
        output[idx] = (input[idx] - total_mean) * inv_std;
    }
}

// Host launcher for group_norm
extern "C" void group_norm_launcher(
    const float* input,
    float* output,
    int N, int G, int Cg, int H, int W, int C,
    float eps,
    cudaStream_t stream)
{
    const int blockSize = 256;
    dim3 grid(G, N); // group index x, batch index y
    int shared_mem_size = 3 * blockSize * sizeof(float);
    group_norm_kernel<<<grid, blockSize, shared_mem_size, stream>>>(
        input, output, N, G, Cg, H, W, C, eps);
}
