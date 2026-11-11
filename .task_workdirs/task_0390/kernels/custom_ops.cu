#include <cuda_runtime.h>

// channel_shuffle forward kernel
// x: input tensor with arbitrary strides (float32)
// output: output tensor contiguous, same shape as x
// N, C, H, W: dimensions
// groups: number of groups for channel shuffle
// strides: input strides for [N, C, H, W] dimensions in elements
__global__ void channel_shuffle_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    int N, int C, int H, int W, int groups,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w)
{
    int channels_per_group = C / groups;

    // Total number of elements
    long long total = (long long)N * C * H * W;

    // Grid-stride loop over flattened index
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x)
    {
        // Decompose idx into (n, c, h, w) in row-major order of output
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int out_c = tmp % C;
        int n = tmp / C;

        // Compute channel shuffle mapping
        // out_c = out_g * channels_per_group + out_cg
        int out_g = out_c / channels_per_group;
        int out_cg = out_c - out_g * channels_per_group;
        // input channel index c = out_cg * groups + out_g
        int in_c = out_cg * groups + out_g;

        // Input linear offset using input strides
        long long in_offset = (long long)n * stride_n + (long long)in_c * stride_c + (long long)h * stride_h + (long long)w * stride_w;
        // Output linear offset (output is contiguous, assumed standard shape stride)
        long long out_offset = (long long)n * (C * H * W) + (long long)out_c * (H * W) + (long long)h * W + w;

        output[out_offset] = x[in_offset];
    }
}

extern "C" void channel_shuffle_forward_launcher(
    const float* x,
    float* output,
    int N, int C, int H, int W, int groups,
    long long stride_n, long long stride_c, long long stride_h, long long stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H * W;
    int block_size = 256;
    // Cap grid to reasonable size and use grid-stride loop
    long long grid_size_ll = (total + block_size - 1) / block_size;
    int max_grid = 65535; // safe for all GPUs, grid-stride loop handles the rest
    int grid_size = (int)min(grid_size_ll, (long long)max_grid);

    channel_shuffle_kernel<<<grid_size, block_size, 0, stream>>>(
        x, output, N, C, H, W, groups,
        stride_n, stride_c, stride_h, stride_w);
}
