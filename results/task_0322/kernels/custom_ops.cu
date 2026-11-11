#include <cuda_runtime.h>

// Channel shuffle kernel: input shape (N, C, H, W), groups g.
// Reshape to (N, g, C/g, H, W), transpose(1,2), reshape back to (N, C, H, W).
// For output element (n, co, h, w):
//   group_idx = co % groups;          // which group in output (because after transpose, dimension 2 (C/g) becomes the fast-varying channel dim)
//   idx_in_group = co / groups;       // position within that group
//   ci = group_idx * channels_per_group + idx_in_group;  // source channel in input
__global__ void channel_shuffle_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W, int groups)
{
    int total = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int channels_per_group = C / groups;
    for (int pos = idx; pos < total; pos += stride) {
        // decompose pos into (n, co, hw)
        int hw = H * W;
        int n = pos / (C * hw);
        int rem = pos - n * C * hw;
        int co = rem / hw;
        int hw_idx = rem - co * hw;

        int group_idx = co % groups;
        int idx_in_group = co / groups;
        int ci = group_idx * channels_per_group + idx_in_group;

        int src_pos = n * C * hw + ci * hw + hw_idx;
        output[pos] = input[src_pos];
    }
}

extern "C" void channel_shuffle_cuda_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W, int groups,
    cudaStream_t stream)
{
    int total = N * C * H * W;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid stride loop handles rest
    channel_shuffle_kernel<<<blocks, threads, 0, stream>>>(
        output, input, N, C, H, W, groups);
}