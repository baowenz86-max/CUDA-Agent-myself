// custom_ops.cu
// This file contains all custom CUDA kernels and launchers.
// Operators: custom_histc, custom_conv_transpose1d (pure CUDA kernel implementation
// because cuDNN headers are unavailable in this environment).

#include <cuda_runtime.h>
#include <math.h>

// ------------------------- custom_histc -------------------------

__global__ void custom_histc_kernel(
    const float* __restrict__ input,
    size_t numel,
    int bins,
    float min_val,
    float max_val,
    float* __restrict__ output)
{
    extern __shared__ unsigned int block_hist[];
    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        block_hist[i] = 0u;
    }
    __syncthreads();

    float bin_width = (max_val - min_val) / bins;
    if (bin_width <= 0.0f) {
        return;
    }

    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < numel; idx += (size_t)gridDim.x * blockDim.x) {
        float val = input[idx];
        if (isnan(val) || isinf(val)) continue;
        if (val < min_val || val > max_val) continue;

        int bin = (int)((val - min_val) / bin_width);
        bin = min(bin, bins - 1);
        bin = max(bin, 0);
        atomicAdd(&block_hist[bin], 1u);
    }

    __syncthreads();

    for (int i = threadIdx.x; i < bins; i += blockDim.x) {
        if (block_hist[i] != 0) {
            atomicAdd(&output[i], (float)block_hist[i]);
        }
    }
}

extern "C" void custom_histc_launcher(
    const float* input,
    size_t numel,
    int bins,
    float min_val,
    float max_val,
    float* output,
    cudaStream_t stream)
{
    if (bins <= 0 || numel == 0) {
        return;
    }

    const int block_size = 256;
    int grid_size = (int)((numel + block_size - 1) / block_size);
    const int max_grid_size = 65535;
    if (grid_size > max_grid_size) grid_size = max_grid_size;
    if (grid_size < 1) grid_size = 1;

    size_t shared_mem = bins * sizeof(unsigned int);

    custom_histc_kernel<<<grid_size, block_size, shared_mem, stream>>>(
        input, numel, bins, min_val, max_val, output);
}

// ------------------------- custom_conv_transpose1d (pure CUDA) -------------------------

__global__ void custom_conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * out_channels * output_length;
    if (idx >= total_elements) return;

    // Decompose idx into (b, oc, ol)
    int b = idx / (out_channels * output_length);
    int remainder = idx % (out_channels * output_length);
    int oc = remainder / output_length;
    int ol = remainder % output_length;

    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int input_pos = (ol + padding - k) / stride;
            if ((ol + padding - k) >= 0 && ((ol + padding - k) % stride == 0) && input_pos < input_length && input_pos >= 0) {
                int input_idx = b * (in_channels * input_length) + ic * input_length + input_pos;
                int weight_idx = ic * (out_channels * kernel_size) + oc * kernel_size + k;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    // Add bias
    sum += bias[oc];
    output[idx] = sum;
}

extern "C" void custom_conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream)
{
    int total_elements = batch * out_channels * output_length;
    const int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    custom_conv_transpose1d_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weight, bias, output,
        batch, in_channels, out_channels, input_length, output_length,
        kernel_size, stride, padding, output_padding);
}
