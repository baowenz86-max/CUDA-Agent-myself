// CUDA kernels for average pooling and tanh
#include <cuda_runtime.h>
#include <math.h>

// AvgPool2d kernel: count_include_pad=True (padding zeros included in average)
// Input: NCHW float32, contiguous
// Output: NCHW float32, same layout
__global__ void avgpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_height * out_width;
    int stride_grid = blockDim.x * gridDim.x;

    int divisor = kernel_size * kernel_size; // count_include_pad=True: denominator is always full kernel area

    for (int linear = idx; linear < total; linear += stride_grid) {
        // Decompose linear index to (n, c, oh, ow)
        int ow = linear % out_width;
        int tmp = linear / out_width;
        int oh = tmp % out_height;
        tmp /= out_height;
        int c = tmp % channels;
        int n = tmp / channels;

        // Compute input window start (including padding)
        int h_start = oh * stride - padding;
        int w_start = ow * stride - padding;

        float sum = 0.0f;

        for (int kh = 0; kh < kernel_size; ++kh) {
            int ih = h_start + kh;
            if (ih < 0 || ih >= in_height) {
                // padding zero contributes 0 to sum; no need to increment count because divisor is fixed
                continue;
            }
            for (int kw = 0; kw < kernel_size; ++kw) {
                int iw = w_start + kw;
                if (iw < 0 || iw >= in_width) {
                    continue;
                }
                int input_idx = ((n * channels + c) * in_height + ih) * in_width + iw;
                sum += input[input_idx];
            }
        }

        // Average including padding (divisor = kernel_size^2)
        output[linear] = sum / static_cast<float>(divisor);
    }
}

// Tanh elementwise kernel
__global__ void tanh_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_grid = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride_grid) {
        output[i] = tanhf(input[i]);
    }
}

// Launcher for avgpool2d_forward
extern "C" void avgpool2d_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream)
{
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    int total = batch * channels * out_height * out_width;

    // Launch configuration: use enough blocks to cover all elements, with grid-stride loop
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    // Cap grid size to avoid excessive blocks (limit to 65535 for older GPUs)
    if (grid_size > 65535) grid_size = 65535;

    avgpool2d_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, batch, channels, in_height, in_width, out_height, out_width,
        kernel_size, stride, padding);
}

// Launcher for tanh_forward
extern "C" void tanh_forward_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    tanh_kernel<<<grid_size, block_size, 0, stream>>>(input, output, total);
}