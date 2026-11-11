#include <cuda_runtime.h>

// CUDA kernels and launchers for ModelNew custom ops.
// Contract: only float32 tensors on CUDA; output bool for isin, float32 for maxpool.
// No PyTorch/ATen headers. All tensors passed as raw pointers plus sizes/strides.

// isin kernel: elementwise check if x value belongs to values (1D contiguous).
__global__ void isin_kernel(
    const float* __restrict__ x,
    const float* __restrict__ values,
    bool* __restrict__ out,
    long long total_elements,
    int values_size)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += stride) {
        float val = x[idx];
        bool found = false;
        for (int v = 0; v < values_size; ++v) {
            if (val == values[v]) {
                found = true;
                break;
            }
        }
        out[idx] = found;
    }
}

// Generic strided isin kernel: pass arrays for shape and strides.
__global__ void isin_kernel_generic(
    const float* __restrict__ x,
    const float* __restrict__ values,
    bool* __restrict__ out,
    long long total_elements,
    int values_size,
    const long long* __restrict__ x_shape,
    const long long* __restrict__ x_strides,
    int dim)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long grid_stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += grid_stride) {
        // Compute multi-index from linear idx (row-major order)
        long long offset = 0;
        long long remaining = idx;
        for (int d = dim - 1; d >= 0; --d) {
            long long coord = remaining % x_shape[d];
            remaining /= x_shape[d];
            offset += coord * x_strides[d];
        }
        float val = x[offset];
        bool found = false;
        for (int v = 0; v < values_size; ++v) {
            if (val == values[v]) {
                found = true;
                break;
            }
        }
        out[idx] = found;  // output is contiguous bool (binding allocates contiguous with x's shape)
    }
}

// Final kernel for isin (handles arbitrary strides for both x and out, but out shape matches x shape).
__global__ void isin_kernel_strided_both(
    const float* __restrict__ x,
    const float* __restrict__ values,
    bool* __restrict__ out,
    long long total_elements,
    int values_size,
    const long long* __restrict__ x_shape,
    const long long* __restrict__ x_strides,
    const long long* __restrict__ out_strides,
    int dim)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long grid_stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += grid_stride) {
        long long remaining = idx;
        long long x_offset = 0;
        long long out_offset = 0;
        for (int d = dim - 1; d >= 0; --d) {
            long long coord = remaining % x_shape[d];
            remaining /= x_shape[d];
            x_offset += coord * x_strides[d];
            out_offset += coord * out_strides[d];
        }
        float val = x[x_offset];
        bool found = false;
        for (int v = 0; v < values_size; ++v) {
            if (val == values[v]) {
                found = true;
                break;
            }
        }
        out[out_offset] = found;
    }
}

// MaxPool1d kernel: operates on last dimension, for each output position, find max in window.
// Input x: shape [batch, channel, length], can be non-contiguous.
// Output y: shape [batch, channel, out_length], contiguous (contract says output is contiguous).
__global__ void maxpool1d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    long long x_batch_stride,
    long long x_channel_stride,
    long long x_length_stride)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long grid_stride = (long long)gridDim.x * blockDim.x;
    long long total_outputs = (long long)batch_size * channels * output_length;
    for (; idx < total_outputs; idx += grid_stride) {
        int out_i = idx % output_length;
        long long tmp = idx / output_length;
        int out_c = tmp % channels;
        int out_b = tmp / channels;

        int start = out_i * stride - padding;
        float max_val = -3.402823466e+38F;
        bool valid = false;
        for (int k = 0; k < kernel_size; ++k) {
            int in_pos = start + k * dilation;
            if (in_pos >= 0 && in_pos < input_length) {
                long long in_offset = (long long)out_b * x_batch_stride + (long long)out_c * x_channel_stride + (long long)in_pos * x_length_stride;
                float val = x[in_offset];
                if (!valid || val > max_val) {
                    max_val = val;
                    valid = true;
                }
            }
        }
        y[idx] = max_val;
    }
}

// Launcher for isin_custom (single extern "C" definition)
extern "C" void isin_custom_launcher(
    const float* x,
    const float* values,
    bool* out,
    const long long* x_shape,
    const long long* x_strides,
    const long long* out_strides,
    int dim,
    int values_size,
    long long total_elements,
    cudaStream_t stream)
{
    int threads = 256;
    long long blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    
    isin_kernel_strided_both<<<(int)blocks, threads, 0, stream>>>(
        x, values, out, total_elements, values_size,
        x_shape, x_strides, out_strides, dim);
}

// Launcher for maxpool1d_custom
extern "C" void maxpool1d_custom_launcher(
    const float* x,
    float* y,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    long long x_batch_stride,
    long long x_channel_stride,
    long long x_length_stride,
    cudaStream_t stream)
{
    long long total_outputs = (long long)batch_size * channels * output_length;
    int threads = 256;
    long long blocks = (total_outputs + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    maxpool1d_kernel<<<(int)blocks, threads, 0, stream>>>(
        x, y, batch_size, channels, input_length, output_length,
        kernel_size, stride, padding, dilation,
        x_batch_stride, x_channel_stride, x_length_stride);
}
