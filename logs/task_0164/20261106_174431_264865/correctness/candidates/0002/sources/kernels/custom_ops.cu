// custom_ops.cu
// CUDA kernels and launchers for maxpool3d and prod_dim operations.
// No cuDNN dependency; maxpool3d is implemented as a custom kernel.

#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// maxpool3d_forward: 3D max pooling kernel
// -----------------------------------------------------------------------------
// Input: NCDHW format, float32, possibly non-contiguous (binding ensures contiguous)
// Output: NCDHW format, float32, allocated by binding
// Parameters: kernel_size, stride, padding, dilation (assumed same for all dims)
// ceil_mode is ignored (only False is supported per contract)

__global__ void maxpool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_depth,
    int input_height,
    int input_width,
    int output_depth,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation)
{
    // Total number of output elements: N*C*D_out*H_out*W_out
    long long total_output = (long long)batch_size * channels * output_depth * output_height * output_width;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long grid_size = (long long)gridDim.x * blockDim.x;

    for (; idx < total_output; idx += grid_size) {
        // Decompose idx into n, c, od, oh, ow
        int ow = idx % output_width;
        long long tmp = idx / output_width;
        int oh = tmp % output_height;
        tmp /= output_height;
        int od = tmp % output_depth;
        tmp /= output_depth;
        int c = tmp % channels;
        int n = tmp / channels;

        // Compute input start indices for pooling window
        int in_d_start = od * stride - padding;
        int in_h_start = oh * stride - padding;
        int in_w_start = ow * stride - padding;

        float max_val = -1e30f; // very negative value
        for (int kd = 0; kd < kernel_size; ++kd) {
            int id = in_d_start + kd * dilation;
            if (id < 0 || id >= input_depth) continue;
            for (int kh = 0; kh < kernel_size; ++kh) {
                int ih = in_h_start + kh * dilation;
                if (ih < 0 || ih >= input_height) continue;
                for (int kw = 0; kw < kernel_size; ++kw) {
                    int iw = in_w_start + kw * dilation;
                    if (iw < 0 || iw >= input_width) continue;
                    // Input index in NCDHW layout
                    long long input_idx = (((long long)n * channels + c) * input_depth
                                         + id) * input_height * input_width
                                         + ih * input_width + iw;
                    float val = input[input_idx];
                    if (val > max_val) max_val = val;
                }
            }
        }

        // Output index: same ordering N,C,D_out,H_out,W_out
        // Since output is contiguous, idx directly maps to output index
        output[idx] = max_val;
    }
}

extern "C" void maxpool3d_forward_launcher(
    const float* input,
    float* output,
    int batch_size,
    int in_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int ceil_mode,
    cudaStream_t stream)
{
    // Compute output dimensions (floor formula, same as PyTorch)
    int output_depth = (input_depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    // Total output elements
    long long total_output = (long long)batch_size * in_channels * output_depth * output_height * output_width;

    // Configure grid and block
    int threads = 256;
    int blocks = (int)((total_output + threads - 1) / threads);
    if (blocks > 65535) {
        // Use max grid dimension for x (2^31-1 in CUDA 9+)
        blocks = 65535; // cap to avoid overflow; grid-stride loop handles rest
    }
    // Actually grid dimension can be larger, but for safety we use a reasonable cap
    // For this task total_output ~ 55M, blocks ~ 215k, which exceeds 65535 for dim x.
    // We can use dim3 grid with y dimension if needed, but simplest: use a 1D grid with max blocks.
    // However, CUDA supports gridDim.x up to 2^31-1, so 65535 is not the actual limit. It's 2147483647.
    // So we can set blocks as calculated, no capping needed (but ensure int doesn't overflow).
    // Since total_output fits in 64-bit, blocks fits in int for this size.

    maxpool3d_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        batch_size, in_channels,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width,
        kernel_size, stride, padding, dilation);
}

// -----------------------------------------------------------------------------
// prod_dim_forward: product along dimension 1 (channels) with keepdim=True
// -----------------------------------------------------------------------------
__global__ void prod_dim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width)
{
    long long total_output = (long long)batch_size * depth * height * width;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long grid_size = (long long)gridDim.x * blockDim.x;

    for (; idx < total_output; idx += grid_size) {
        int w = idx % width;
        long long tmp = idx / width;
        int h = tmp % height;
        tmp /= height;
        int d = tmp % depth;
        int b = tmp / depth;

        // Input base index for (b, 0, d, h, w) assuming contiguous NCDHW
        long long base = ((long long)b * channels) * depth * height * width
                       + d * height * width + h * width + w;

        float prod = 1.0f;
        for (int c = 0; c < channels; ++c) {
            prod *= input[base + (long long)c * depth * height * width];
        }

        // Output index: b * depth * height * width + d * height * width + h * width + w
        long long out_idx = (long long)b * depth * height * width
                          + (long long)d * height * width + (long long)h * width + w;
        output[out_idx] = prod;
    }
}

extern "C" void prod_dim_forward_launcher(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    cudaStream_t stream)
{
    long long total_output = (long long)batch_size * depth * height * width;
    int threads = 256;
    int blocks = (int)((total_output + threads - 1) / threads);

    prod_dim_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch_size, channels, depth, height, width);
}
