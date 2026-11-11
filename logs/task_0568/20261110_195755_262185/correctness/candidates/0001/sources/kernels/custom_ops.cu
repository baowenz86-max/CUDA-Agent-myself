// CUDA kernels and launchers for model_new.py custom ops
// This file contains only CUDA C++ code (no torch/ATen/pybind dependencies)
// All launchers use raw pointers, scalar parameters, and cudaStream_t.
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

// -----------------------------------------------------------------------------
// custom_pad
// Reference: x = self.pad(x) with ConstantPad2d(padding=(left,right,top,bottom), value=value)
// Input: 4D tensor [N, C, H, W] (here C=1), float32, arbitrary strides
// Output: 4D contiguous tensor [N, C, H+top+bottom, W+left+right]
// The kernel writes one output element per thread, using grid-stride loop.
// Boundary regions are filled with the constant value; interior is copied from input.
// -----------------------------------------------------------------------------
__global__ void custom_pad_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    float pad_value,
    long long input_n_stride, long long input_c_stride,
    long long input_h_stride, long long input_w_stride)
{
    long long total = (long long)N * C * H_out * W_out;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x)
    {
        int w = idx % W_out;
        long long tmp = idx / W_out;
        int h = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        int h_in = h - pad_top;
        int w_in = w - pad_left;

        float val;
        if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
            // Inside original input region
            const float* src = input + n * input_n_stride + c * input_c_stride
                               + h_in * input_h_stride + w_in * input_w_stride;
            val = *src;
        } else {
            val = pad_value;
        }
        output[idx] = val;
    }
}

extern "C" void custom_pad_launcher(
    float* output,
    const float* input,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    float pad_value,
    long long input_n_stride, long long input_c_stride,
    long long input_h_stride, long long input_w_stride,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H_out * W_out;
    if (total == 0) return;

    int block_size = 256;
    long long grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL; // grid size limit

    custom_pad_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        output, input,
        N, C, H_in, W_in, H_out, W_out,
        pad_left, pad_right, pad_top, pad_bottom,
        pad_value,
        input_n_stride, input_c_stride, input_h_stride, input_w_stride);
}

// -----------------------------------------------------------------------------
// custom_max_dim1
// Reference: x = torch.max(x, dim=1)[0]
// Input: 4D tensor [N, 1, H, W] (dim=1 size is 1), float32, arbitrary strides
// Output: 3D contiguous tensor [N, H, W], containing same values (since max over size-1 dim is identity)
// This is effectively a squeeze along dim=1.
// -----------------------------------------------------------------------------
__global__ void custom_max_dim1_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int H, int W,
    long long input_n_stride, long long input_c_stride,
    long long input_h_stride, long long input_w_stride)
{
    long long total = (long long)N * H * W;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x)
    {
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        int n = tmp / H;

        // Since C=1, only copy index c=0
        const float* src = input + n * input_n_stride + 0 * input_c_stride
                           + h * input_h_stride + w * input_w_stride;
        output[idx] = *src;
    }
}

extern "C" void custom_max_dim1_launcher(
    float* output,
    const float* input,
    int N, int H, int W,
    long long input_n_stride, long long input_c_stride,
    long long input_h_stride, long long input_w_stride,
    cudaStream_t stream)
{
    long long total = (long long)N * H * W;
    if (total == 0) return;

    int block_size = 256;
    long long grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL;

    custom_max_dim1_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        output, input,
        N, H, W,
        input_n_stride, input_c_stride, input_h_stride, input_w_stride);
}
