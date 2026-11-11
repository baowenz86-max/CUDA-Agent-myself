#include <cuda_runtime.h>
#include <cstdint>

// Kernel: view_as_real
// Input: complex64 tensor of shape [B, H, W] with arbitrary strides (in elements)
// Output: float32 tensor of shape [B, H, W, 2], contiguous, with last dim size 2
// We treat input as float array (each complex is 2 floats)
__global__ void view_as_real_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_complex,
    int64_t stride_b, int64_t stride_h, int64_t stride_w)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_complex; i += stride) {
        // Compute 3D coordinates from flat index assuming contiguous input layout
        int64_t w = i % stride_w;
        int64_t tmp = i / stride_w;
        int64_t h = tmp % stride_h;
        int64_t b = tmp / stride_h;

        // input is float*: each complex number occupies two consecutive floats.
        // The strides are in complex elements, so we multiply by 2 to get float offset.
        int64_t input_offset = (b * stride_b + h * stride_h + w * stride_w) * 2;
        float real = input[input_offset];
        float imag = input[input_offset + 1];
        int64_t out_offset = i * 2;
        output[out_offset] = real;
        output[out_offset + 1] = imag;
    }
}

// Launcher for view_as_real
// input_data: pointer to complex64 elements (as float array)
// output_data: pointer to float32 output, size total_complex*2
// dims: array of 3 int64_t [B, H, W]
// strides: array of 3 int64_t [stride_b, stride_h, stride_w] in complex elements
// stream: CUDA stream
extern "C" void view_as_real_launcher(
    const void* input_data,
    float* output_data,
    const int64_t* dims,
    const int64_t* strides,
    cudaStream_t stream)
{
    int64_t total_complex = dims[0] * dims[1] * dims[2];
    int threads = 256;
    int blocks = (int)((total_complex + threads - 1) / threads);
    view_as_real_kernel<<<blocks, threads, 0, stream>>>(
        (const float*)input_data,
        output_data,
        total_complex,
        strides[0], strides[1], strides[2]
    );
}

// Kernel: constant_pad
// Input: float32 tensor of shape [B, H, W, 2] with arbitrary strides (in elements)
// Output: float32 tensor of shape [B, H, W+2, 4], contiguous, last dim size 4
// Pad: pad_left=1, pad_right=1, pad_top=1, pad_bottom=1 (applied to last two dims: W and H)
// Value: pad_value (float)
__global__ void constant_pad_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t B, int64_t H_in, int64_t W_in,
    int64_t stride_b, int64_t stride_h, int64_t stride_w,
    int64_t out_H, int64_t out_W,
    float pad_value)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    int64_t total_out = B * out_H * out_W * 4;
    for (int64_t i = idx; i < total_out; i += stride) {
        // Compute output coordinates
        int64_t c = i % 4; // last dim size 4
        int64_t tmp = i / 4;
        int64_t w_out = tmp % out_W;
        tmp /= out_W;
        int64_t h_out = tmp % out_H;
        int64_t b = tmp / out_H;

        // Input coordinates: w_in = w_out - 1, h_in = h_out, c_in = c - 1
        int64_t w_in = w_out - 1;
        int64_t h_in = h_out;
        int64_t c_in = c - 1;

        bool is_pad = (w_in < 0 || w_in >= W_in || c_in < 0 || c_in >= 2);
        // h_in is always valid because H_out = H_in.

        if (is_pad) {
            output[i] = pad_value;
        } else {
            // Compute input offset using input strides (in elements)
            // Input shape [B, H_in, W_in, 2], strides [stride_b, stride_h, stride_w, 1]
            // Note: the last dimension stride is assumed 1.
            int64_t offset = b * stride_b + h_in * stride_h + w_in * stride_w + c_in;
            output[i] = input[offset];
        }
    }
}

// Launcher for constant_pad
// input_data: pointer to float32 input [B, H_in, W_in, 2]
// output_data: pointer to float32 output [B, H_out, W_out, 4]
// B, H_in, W_in, strides for first three dims (last dim stride assumed 1)
// pad value as float
// output dims computed from input and pad.
extern "C" void constant_pad_launcher(
    const float* input_data,
    float* output_data,
    int64_t B, int64_t H_in, int64_t W_in,
    int64_t stride_b, int64_t stride_h, int64_t stride_w,
    float pad_value,
    cudaStream_t stream)
{
    int64_t H_out = H_in; // no padding on H dimension
    int64_t W_out = W_in + 2;
    int64_t total_out = B * H_out * W_out * 4;
    int threads = 256;
    int blocks = (int)((total_out + threads - 1) / threads);
    constant_pad_kernel<<<blocks, threads, 0, stream>>>(
        input_data, output_data, B, H_in, W_in,
        stride_b, stride_h, stride_w,
        H_out, W_out, pad_value);
}
