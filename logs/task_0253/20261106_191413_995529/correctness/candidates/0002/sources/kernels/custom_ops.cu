// kernels/custom_ops.cu
// CUDA kernels for custom ops: Conv2d (naive), ReLU, MaxPool2d.
#include <cuda_runtime.h>

// ReLU forward: y = max(x, 0)
__global__ void relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        output[linear] = fmaxf(input[linear], 0.0f);
    }
}

// MaxPool2d forward
__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_height,
    int kernel_width,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total_outputs = (int64_t)batch_size * channels * output_height * output_width;
    int64_t stride_total = (int64_t)gridDim.x * blockDim.x;

    for (int64_t linear = idx; linear < total_outputs; linear += stride_total) {
        int ow = linear % output_width;
        int64_t tmp = linear / output_width;
        int oh = tmp % output_height;
        tmp /= output_height;
        int c = tmp % channels;
        int b = tmp / channels;

        float max_val = -1e30f;
        bool found = false;
        int ih_start = oh * stride_h - padding_h;
        int iw_start = ow * stride_w - padding_w;

        for (int kh = 0; kh < kernel_height; ++kh) {
            int ih = ih_start + kh * dilation_h;
            if (ih < 0 || ih >= input_height) continue;
            for (int kw = 0; kw < kernel_width; ++kw) {
                int iw = iw_start + kw * dilation_w;
                if (iw < 0 || iw >= input_width) continue;
                const float val = input[(((int64_t)b * channels + c) * input_height + ih) * input_width + iw];
                if (!found || val > max_val) {
                    max_val = val;
                    found = true;
                }
            }
        }
        output[(((int64_t)b * channels + c) * output_height + oh) * output_width + ow] = max_val;
    }
}

// Conv2d forward: supports stride, padding, dilation, groups (groups must divide channels)
__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_height,
    int kernel_width,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total_outputs = (int64_t)batch_size * out_channels * out_height * out_width; // out_height/out_width need to be passed or computed? We'll pass them separately.
    int64_t stride_total = (int64_t)gridDim.x * blockDim.x;

    int channels_per_group = in_channels / groups;
    int out_channels_per_group = out_channels / groups;

    for (int64_t linear = idx; linear < total_outputs; linear += stride_total) {
        int ow = linear % out_width;
        int64_t tmp = linear / out_width;
        int oh = tmp % out_height;
        tmp /= out_height;
        int oc = tmp % out_channels;
        int b = tmp / out_channels;

        int group = oc / out_channels_per_group;
        int in_channel_start = group * channels_per_group;
        float sum = 0.0f;
        for (int ic = 0; ic < channels_per_group; ++ic) {
            int in_c = in_channel_start + ic;
            for (int kh = 0; kh < kernel_height; ++kh) {
                int ih = oh * stride_h - padding_h + kh * dilation_h;
                if (ih < 0 || ih >= in_height) continue;
                for (int kw = 0; kw < kernel_width; ++kw) {
                    int iw = ow * stride_w - padding_w + kw * dilation_w;
                    if (iw < 0 || iw >= in_width) continue;
                    float input_val = input[(((int64_t)b * in_channels + in_c) * in_height + ih) * in_width + iw];
                    float weight_val = weight[(((int64_t)oc * channels_per_group + ic) * kernel_height + kh) * kernel_width + kw];
                    sum += input_val * weight_val;
                }
            }
        }
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[(((int64_t)b * out_channels + oc) * out_height + oh) * out_width + ow] = sum;
    }
}

// Launchers
extern "C" void launch_relu_forward(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements);
}

extern "C" void launch_maxpool2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool ceil_mode,
    cudaStream_t stream)
{
    int threads = 256;
    int64_t total_outputs = (int64_t)batch_size * channels * output_height * output_width;
    int blocks = (int)((total_outputs + threads - 1) / threads);
    maxpool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch_size, channels, input_height, input_width,
        output_height, output_width, kernel_size, kernel_size,
        stride, stride, padding, padding, dilation, dilation);
}

extern "C" void launch_conv2d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_height,
    int kernel_width,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int out_height,
    int out_width,
    cudaStream_t stream)
{
    int threads = 256;
    int64_t total_outputs = (int64_t)batch_size * out_channels * out_height * out_width;
    int blocks = (int)((total_outputs + threads - 1) / threads);
    conv2d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch_size, in_channels, in_height, in_width,
        out_channels, kernel_height, kernel_width,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w, groups,
        out_height, out_width);
}
