// CUDA kernels and launchers for custom ops (except ConvTranspose1d which uses cuDNN in C++)
#include <cuda_runtime.h>
#include <math.h>

// Forward declaration for conv_transpose1d_forward launcher, implemented in binding C++ using cuDNN.
// This declaration ensures stable ABI for the next stage.
extern "C" void conv_transpose1d_forward_launcher(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch_size, int in_channels, int out_channels,
    int input_length, int output_length,
    int kernel_size, int stride, int padding, int output_padding,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// add_bias_forward: output[b, c, l] = input[b, c, l] + bias[c]
// input shape: [batch_size, out_channels, length]
// bias shape: [out_channels, 1] (stored contiguous, stride[0]=1, stride[1]=1)
// output: same shape as input, contiguous
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
        // Assume input and output are contiguous in the innermost dimension,
        // but possibly strided between elements if not contiguous? Contract allows strided.
        // Here input_stride/output_stride are element strides; for contiguous they are 1.
        // We pass element index and use pointer arithmetic with stride.
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
// output same shape as input.
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
