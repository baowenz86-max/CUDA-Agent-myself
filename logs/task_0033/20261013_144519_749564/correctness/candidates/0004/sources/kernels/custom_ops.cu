#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// LayerNorm kernel for input shape [batch, out_channels, length_out]
// normalized over the last dimension (length_out).
// weight and bias shapes: [out_channels, length_out] (same as normalized_shape).
// Input, weight, bias all support arbitrary strides.
// -----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void layer_norm_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int batch,
    int channels,
    int length,
    int input_batch_stride,
    int input_channel_stride,
    int weight_channel_stride,
    int weight_inner_stride,
    int bias_channel_stride,
    int bias_inner_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels;
    if (idx >= total) return;

    int b = idx / channels;
    int c = idx - b * channels;

    const float* in_ptr = input + b * input_batch_stride + c * input_channel_stride;
    float* out_ptr = output + (size_t)b * channels * length + (size_t)c * length;

    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        sum += in_ptr[i * input_channel_stride];
    }
    float mean = sum / length;

    // Compute variance (biased)
    float var_sum = 0.0f;
    for (int i = 0; i < length; ++i) {
        float diff = in_ptr[i * input_channel_stride] - mean;
        var_sum += diff * diff;
    }
    float var = var_sum / length;
    float inv_std = rsqrtf(var + eps);

    // Apply normalization and affine
    const float* w_ptr = weight + c * weight_channel_stride;
    const float* b_ptr = bias + c * bias_channel_stride;
    for (int i = 0; i < length; ++i) {
        float val = in_ptr[i * input_channel_stride];
        float norm = (val - mean) * inv_std;
        float w = w_ptr[i * weight_inner_stride];
        float b = b_ptr[i * bias_inner_stride];
        out_ptr[i] = norm * w + b;
    }
}

// -----------------------------------------------------------------------------
// Addcmul kernel: output = input + value * tensor1 * tensor2
// All tensors have same shape [batch, channels, length]
// Supports arbitrary strides.
// -----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void addcmul_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ tensor1,
    const float* __restrict__ tensor2,
    float value,
    int total_elements,
    int batch,
    int channels,
    int length,
    int input_batch_stride,
    int input_channel_stride,
    int tensor1_batch_stride,
    int tensor1_channel_stride,
    int tensor2_batch_stride,
    int tensor2_channel_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int linear = idx; linear < total_elements; linear += stride) {
        int l = linear % length;
        int tmp = linear / length;
        int c = tmp % channels;
        int b = tmp / channels;

        const float* in_ptr = input + b * input_batch_stride + c * input_channel_stride + l;
        const float* t1_ptr = tensor1 + b * tensor1_batch_stride + c * tensor1_channel_stride + l;
        const float* t2_ptr = tensor2 + b * tensor2_batch_stride + c * tensor2_channel_stride + l;
        float* out_ptr = output + (size_t)b * channels * length + (size_t)c * length + l;

        *out_ptr = *in_ptr + value * (*t1_ptr) * (*t2_ptr);
    }
}

// -----------------------------------------------------------------------------
// Launchers (extern "C")
// -----------------------------------------------------------------------------
extern "C" void layer_norm_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    float eps,
    int batch,
    int channels,
    int length,
    int input_batch_stride,
    int input_channel_stride,
    int weight_channel_stride,
    int weight_inner_stride,
    int bias_channel_stride,
    int bias_inner_stride,
    cudaStream_t stream)
{
    const int block_size = 256;
    int total_rows = batch * channels;
    int grid = (total_rows + block_size - 1) / block_size;
    layer_norm_kernel<block_size><<<grid, block_size, 0, stream>>>(
        output, input, weight, bias, eps,
        batch, channels, length,
        input_batch_stride, input_channel_stride,
        weight_channel_stride, weight_inner_stride,
        bias_channel_stride, bias_inner_stride);
}

extern "C" void addcmul_launcher(
    float* output,
    const float* input,
    const float* tensor1,
    const float* tensor2,
    float value,
    int batch,
    int channels,
    int length,
    int input_batch_stride,
    int input_channel_stride,
    int tensor1_batch_stride,
    int tensor1_channel_stride,
    int tensor2_batch_stride,
    int tensor2_channel_stride,
    cudaStream_t stream)
{
    const int block_size = 256;
    int total_elements = batch * channels * length;
    int grid = (total_elements + block_size - 1) / block_size;
    if (grid > 65535) grid = 65535;  // grid-stride loop handles larger
    addcmul_kernel<block_size><<<grid, block_size, 0, stream>>>(
        output, input, tensor1, tensor2, value,
        total_elements, batch, channels, length,
        input_batch_stride, input_channel_stride,
        tensor1_batch_stride, tensor1_channel_stride,
        tensor2_batch_stride, tensor2_channel_stride);
}
