#include <cuda_runtime.h>
#include <math.h>

// Forward declaration for conv_transpose1d_forward_launcher, implemented in custom_ops.cu
// This launcher replaces the cuDNN implementation due to missing cudnn.h.
extern "C" void conv_transpose1d_forward_launcher(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_length, int output_length,
    int kernel_size, int stride, int padding, int output_padding,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// add_bias_forward: output[b, c, l] = input[b, c, l] + bias[c]
// -----------------------------------------------------------------------------
__global__ void add_bias_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int out_channels, int length,
    int input_batch_stride, int input_channel_stride,
    int output_batch_stride, int output_channel_stride)
{
    int total = batch_size * out_channels * length;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int l = idx % length;
        int tmp = idx / length;
        int c = tmp % out_channels;
        int b = tmp / out_channels;
        float val = input[b * input_batch_stride + c * input_channel_stride + l];
        val += bias[c];
        output[b * output_batch_stride + c * output_channel_stride + l] = val;
    }
}

extern "C" void add_bias_forward_launcher(
    const float* input, const float* bias, float* output,
    int batch_size, int out_channels, int length,
    int input_batch_stride, int input_channel_stride,
    int output_batch_stride, int output_channel_stride,
    cudaStream_t stream)
{
    int total = batch_size * out_channels * length;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535;
    add_bias_forward_kernel<<<grid, block, 0, stream>>>(
        input, bias, output,
        batch_size, out_channels, length,
        input_batch_stride, input_channel_stride,
        output_batch_stride, output_channel_stride);
}

// -----------------------------------------------------------------------------
// round_forward: output[i] = torch.round(input[i])
// torch.round rounds half away from zero.
// -----------------------------------------------------------------------------
__device__ inline float round_away_from_zero(float x) {
    float t = truncf(x);
    float diff = x - t;
    if (diff > 0.5f || (diff == 0.5f && x > 0.0f))
        return t + 1.0f;
    else if (diff < -0.5f || (diff == -0.5f && x < 0.0f))
        return t - 1.0f;
    else
        return t;
}

__global__ void round_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total,
    int input_stride, int output_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        float val = input[idx * input_stride];
        output[idx * output_stride] = round_away_from_zero(val);
    }
}

extern "C" void round_forward_launcher(
    const float* input, float* output, int total,
    int input_stride, int output_stride, cudaStream_t stream)
{
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535;
    round_forward_kernel<<<grid, block, 0, stream>>>(
        input, output, total, input_stride, output_stride);
}

// -----------------------------------------------------------------------------
// triu_forward: for each batch b, out_channels x length matrix, set elements where col < row to zero.
// -----------------------------------------------------------------------------
__global__ void triu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int out_channels, int length,
    int input_batch_stride, int input_channel_stride,
    int output_batch_stride, int output_channel_stride)
{
    int total = batch_size * out_channels * length;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int l = idx % length;
        int tmp = idx / length;
        int c = tmp % out_channels;
        int b = tmp / out_channels;
        float val = input[b * input_batch_stride + c * input_channel_stride + l];
        if (l < c) {
            val = 0.0f;
        }
        output[b * output_batch_stride + c * output_channel_stride + l] = val;
    }
}

extern "C" void triu_forward_launcher(
    const float* input, float* output,
    int batch_size, int out_channels, int length,
    int input_batch_stride, int input_channel_stride,
    int output_batch_stride, int output_channel_stride,
    cudaStream_t stream)
{
    int total = batch_size * out_channels * length;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535;
    triu_forward_kernel<<<grid, block, 0, stream>>>(
        input, output,
        batch_size, out_channels, length,
        input_batch_stride, input_channel_stride,
        output_batch_stride, output_channel_stride);
}

// -----------------------------------------------------------------------------
// conv_transpose1d_forward_kernel: 朴素实现，每个输出位置累加所有相关输入
// 输入形状: [batch_size, in_channels, input_length]
// 权重形状: [in_channels, out_channels, kernel_size]
// 输出形状: [batch_size, out_channels, output_length]
// 公式：output[b, oc, ol] = sum_{ic, k} input[b, ic, il] * weight[ic, oc, k]
// 其中 il = (ol + padding - k) / stride，且 (ol + padding - k) % stride == 0，0 <= il < input_length
// 同时要加上 bias（如果有），但 bias 已在 binding 中处理，这里不处理。
// -----------------------------------------------------------------------------
__global__ void conv_transpose1d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size, int in_channels, int out_channels,
    int input_length, int output_length,
    int kernel_size, int stride, int padding, int output_padding)
{
    int total = batch_size * out_channels * output_length;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int global_stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += global_stride) {
        int ol = idx % output_length;
        int tmp = idx / output_length;
        int oc = tmp % out_channels;
        int b = tmp / out_channels;

        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            const float* input_ptr = input + b * in_channels * input_length + ic * input_length;
            const float* weight_ptr = weight + ic * out_channels * kernel_size + oc * kernel_size;
            for (int k = 0; k < kernel_size; ++k) {
                int numerator = ol + padding - k;
                if (numerator < 0) continue;
                int il = numerator / stride;
                if (il >= input_length) continue;
                if (numerator % stride != 0) continue;
                sum += input_ptr[il] * weight_ptr[k];
            }
        }
        // output_padding 不影响计算，只影响输出尺寸，已在调用前确定 output_length
        output[b * out_channels * output_length + oc * output_length + ol] = sum;
    }
}

extern "C" void conv_transpose1d_forward_launcher(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_length, int output_length,
    int kernel_size, int stride, int padding, int output_padding,
    cudaStream_t stream)
{
    int total = batch_size * out_channels * output_length;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535;
    conv_transpose1d_forward_kernel<<<grid, block, 0, stream>>>(
        input, weight, output,
        batch_size, in_channels, out_channels,
        input_length, output_length,
        kernel_size, stride, padding, output_padding);
}
