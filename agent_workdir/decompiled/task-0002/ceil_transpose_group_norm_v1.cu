#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>

// Best-effort reconstruction from the supplied SASS.
// The assembly is low-level and does not preserve the original high-level
// source exactly, so this file reflects the likely kernel intent:
// a fused group-normalization kernel operating on an N x C x H x W tensor.
//
// The kernel computes, for every (n, h, w) location, the mean and variance
// across all channels in the same group, then applies:
//   y = (x - mean) / sqrt(var + eps)
//
// The original SASS suggests a highly optimized version with transposed /
// packed memory access patterns, but this CUDA version keeps the semantics
// clear and compiles as a valid implementation.

static constexpr float kEps = 1e-5f;

template <int kBlockSize>
__global__ void fused_group_norm_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int n,
                                        int c,
                                        int h,
                                        int w,
                                        int groups) {
    const int total = n * c * h * w;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) {
        return;
    }

    const int hw = h * w;
    const int c_per_group = std::max(1, (c + groups - 1) / groups);

    const int pos = tid;
    const int n_idx = pos / (c * hw);
    const int rem = pos % (c * hw);
    const int ch_idx = rem / hw;
    const int hw_idx = rem % hw;

    const int group_id = ch_idx / c_per_group;
    const int group_start = group_id * c_per_group;
    const int group_end = std::min(c, group_start + c_per_group);

    float sum = 0.0f;
    float sum_sq = 0.0f;

    for (int ch = group_start; ch < group_end; ++ch) {
        const int linear = ((n_idx * c + ch) * hw + hw_idx);
        const float x = input[linear];
        sum += x;
        sum_sq += x * x;
    }

    const float mean = sum / static_cast<float>(group_end - group_start);
    const float var = sum_sq / static_cast<float>(group_end - group_start) - mean * mean;
    const float inv_std = rsqrtf(var + kEps);

    const int input_idx = ((n_idx * c + ch_idx) * hw + hw_idx);
    output[input_idx] = (input[input_idx] - mean) * inv_std;
}

void launch_fused_group_norm_kernel(const float* d_input,
                                    float* d_output,
                                    int n,
                                    int c,
                                    int h,
                                    int w,
                                    int groups,
                                    cudaStream_t stream = 0) {
    const int total = n * c * h * w;
    const int block_size = 256;
    const int grid_size = (total + block_size - 1) / block_size;

    fused_group_norm_kernel<block_size><<<grid_size, block_size, 0, stream>>>(
        d_input, d_output, n, c, h, w, groups);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "launch_fused_group_norm_kernel failed: %s\n", cudaGetErrorString(err));
    }
}
