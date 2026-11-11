// custom_ops.cu
// CUDA kernels and launchers for maxpool3d and prod_dim operations.

#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// maxpool3d_forward: 3D max pooling kernel
// -----------------------------------------------------------------------------
// Input: NCDHW format, float32, contiguous (binding ensures contiguous)
// Output: NCDHW format, float32, allocated by binding
// Parameters: kernel_size, stride, padding, dilation (per dimension)
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
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w)
{
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

        int in_d_start = od * stride_d - padding_d;
        int in_h_start = oh * stride_h - padding_h;
        int in_w_start = ow * stride_w - padding_w;

        float max_val = -1e30f;
        for (int kd = 0; kd < kernel_d; ++kd) {
            int id = in_d_start + kd * dilation_d;
            if (id < 0 || id >= input_depth) continue;
            for (int kh = 0; kh < kernel_h; ++kh) {
                int ih = in_h_start + kh * dilation_h;
                if (ih < 0 || ih >= input_height) continue;
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int iw = in_w_start + kw * dilation_w;
                    if (iw < 0 || iw >= input_width) continue;
                    long long input_idx = (((long long)n * channels + c) * input_depth
                                         + id) * input_height * input_width
                                         + ih * input_width + iw;
                    float val = input[input_idx];
                    if (val > max_val) max_val = val;
                }
            }
        }

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
    int kernel_d, int kernel_h, int kernel_w,
    int stride_d, int stride_h, int stride_w,
    int padding_d, int padding_h, int padding_w,
    int dilation_d, int dilation_h, int dilation_w,
    cudaStream_t stream)
{
    int output_depth = (input_depth + 2 * padding_d - dilation_d * (kernel_d - 1) - 1) / stride_d + 1;
    int output_height = (input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int output_width = (input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    long long total_output = (long long)batch_size * in_channels * output_depth * output_height * output_width;

    int threads = 256;
    int blocks = (int)((total_output + threads - 1) / threads);

    maxpool3d_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        batch_size, in_channels,
        input_depth, input_height, input_width,
        output_depth, output_height, output_width,
        kernel_d, kernel_h, kernel_w,
        stride_d, stride_h, stride_w,
        padding_d, padding_h, padding_w,
        dilation_d, dilation_h, dilation_w);
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

        long long base = ((long long)b * channels) * depth * height * width
                       + d * height * width + h * width + w;

        float prod = 1.0f;
        for (int c = 0; c < channels; ++c) {
            prod *= input[base + (long long)c * depth * height * width];
        }

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
