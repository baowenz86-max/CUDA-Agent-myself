#include <cuda_runtime.h>

// Channel shuffle kernel: input shape (N, C, H, W), groups g.
// Reshape to (N, g, C/g, H, W), transpose(1,2), reshape back to (N, C, H, W).
// For output element (n, co, h, w):
//   g_idx = co / (C/g);   // which group in output (0..g-1)
//   c_in_group = co % (C/g); // channel within group
//   ci = c_in_group * g + g_idx; // source channel index
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

        int g_idx = co / channels_per_group;
        int c_in_group = co - g_idx * channels_per_group;
        int ci = c_in_group * groups + g_idx;

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
