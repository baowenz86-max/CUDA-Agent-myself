#include <cuda_runtime.h>

// CUDA kernels and launchers for ModelNew custom ops.
// Contract: only float32 tensors on CUDA; output bool for isin, float32 for maxpool.
// No PyTorch/ATen headers. All tensors passed as raw pointers plus sizes/strides.

// isin kernel for 3D tensors (batch, channel, length) with arbitrary strides.
__global__ void isin_kernel_3d(
    const float* __restrict__ x,
    const float* __restrict__ values,
    bool* __restrict__ out,
    long long total_elements,
    int values_size,
    long long shape0, long long shape1, long long shape2,
    long long x_stride0, long long x_stride1, long long x_stride2,
    long long out_stride0, long long out_stride1, long long out_stride2)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long grid_stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += grid_stride) {
        // Compute 3D multi-index from linear idx (row-major order)
        long long coord0 = idx / (shape1 * shape2);
        long long rem = idx % (shape1 * shape2);
        long long coord1 = rem / shape2;
        long long coord2 = rem % shape2;
        long long x_offset = coord0 * x_stride0 + coord1 * x_stride1 + coord2 * x_stride2;
        long long out_offset = coord0 * out_stride0 + coord1 * out_stride1 + coord2 * out_stride2;
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
        int out_i = (int)(idx % output_length);
        long long tmp = idx / output_length;
        int out_c = (int)(tmp % channels);
        int out_b = (int)(tmp / channels);

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

// Launcher for isin_custom: receives shape and strides as scalars (3D).
// No dynamic memory on host that could be freed before async kernel finishes.
extern "C" void isin_custom_launcher(
    const float* x,
    const float* values,
    bool* out,
    long long shape0, long long shape1, long long shape2,
    long long x_stride0, long long x_stride1, long long x_stride2,
    long long out_stride0, long long out_stride1, long long out_stride2,
    int values_size,
    long long total_elements,
    cudaStream_t stream)
{
    int threads = 256;
    long long blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    
    isin_kernel_3d<<<(int)blocks, threads, 0, stream>>>(
        x, values, out, total_elements, values_size,
        shape0, shape1, shape2,
        x_stride0, x_stride1, x_stride2,
        out_stride0, out_stride1, out_stride2);
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
