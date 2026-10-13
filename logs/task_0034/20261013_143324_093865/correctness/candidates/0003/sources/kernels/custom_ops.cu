#include <cuda_runtime.h>

// CUDA kernel for nearest neighbor upsample
__global__ void upsample_nearest_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int channels,
    int H_in,
    int W_in,
    int scale_factor,
    int H_out,
    int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int c = tmp % channels;
    int b = tmp / channels;

    int h_in = h_out / scale_factor;
    int w_in = w_out / scale_factor;

    int input_idx = ((b * channels + c) * H_in + h_in) * W_in + w_in;
    output[idx] = input[input_idx];
}

// CUDA kernel for one-hot encoding
__global__ void one_hot_kernel(
    const int64_t* __restrict__ input,
    int64_t* __restrict__ output,
    int total_elements,
    int num_classes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int64_t label = input[idx];
    int64_t* row = output + idx * num_classes;
    for (int k = 0; k < num_classes; ++k) {
        row[k] = (k == label) ? 1 : 0;
    }
}

// CUDA kernel for transposed convolution (ConvTranspose2d)
// Weight shape: [in_channels, out_channels, kernel_size, kernel_size]
// Input: [batch, in_channels, H_in, W_in]
// Output: [batch, out_channels, H_out, W_out]
__global__ void conv_transpose_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int H_in,
    int W_in,
    int H_out,
    int W_out,
    int kernel_size,
    int stride,
    int padding,
    int output_padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int oc = tmp % out_channels;
    int b = tmp / out_channels;

    float value = 0.0f;
    // Iterate over input channels and kernel positions
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_size; ++kh) {
            int h_in = (h_out + padding - kh) / stride;
            if ((h_out + padding - kh) % stride != 0) continue;
            if (h_in < 0 || h_in >= H_in) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int w_in = (w_out + padding - kw) / stride;
                if ((w_out + padding - kw) % stride != 0) continue;
                if (w_in < 0 || w_in >= W_in) continue;

                int input_idx = ((b * in_channels + ic) * H_in + h_in) * W_in + w_in;
                // Weight layout: [in_channels, out_channels, kh, kw]
                int weight_idx = ((ic * out_channels + oc) * kernel_size + kh) * kernel_size + kw;
                value += input[input_idx] * weight[weight_idx];
            }
        }
    }
    if (bias != nullptr) {
        value += bias[oc];
    }
    output[idx] = value;
}

// Launcher for conv_transpose using custom kernel
extern "C" void conv_transpose_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int H_in,
    int W_in,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream)
{
    int H_out = (H_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + kernel_size + output_padding;

    int total = batch * out_channels * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    conv_transpose_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch, in_channels, out_channels, H_in, W_in, H_out, W_out,
        kernel_size, stride, padding, output_padding);
}

// Launcher for upsample_nearest
extern "C" void upsample_nearest_forward(
    const float* input,
    float* output,
    int batch,
    int channels,
    int H_in,
    int W_in,
    int scale_factor,
    cudaStream_t stream)
{
    int H_out = H_in * scale_factor;
    int W_out = W_in * scale_factor;
    int total = batch * channels * H_out * W_out;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    upsample_nearest_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch, channels, H_in, W_in, scale_factor, H_out, W_out);
}

// Launcher for one_hot
extern "C" void one_hot_forward(
    const int64_t* input,
    int64_t* output,
    int total_elements,
    int num_classes,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    one_hot_kernel<<<blocks, threads, 0, stream>>>(
        input, output, total_elements, num_classes);
}
