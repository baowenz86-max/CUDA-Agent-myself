#include <cuda_runtime.h>

// Leaky ReLU kernel: supports arbitrary input strides.
__global__ void leaky_relu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int n, int c, int h, int w,
    int in_n_stride, int in_c_stride, int in_h_stride,
    float negative_slope)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * h * w;
    if (idx >= total) return;

    // Compute 4D indices from linear index
    int rem = idx;
    int n_idx = rem / (c * h * w);
    rem -= n_idx * (c * h * w);
    int c_idx = rem / (h * w);
    rem -= c_idx * (h * w);
    int h_idx = rem / w;
    int w_idx = rem - h_idx * w;

    int in_offset = n_idx * in_n_stride + c_idx * in_c_stride + h_idx * in_h_stride + w_idx;
    float val = input[in_offset];
    output[idx] = val >= 0.0f ? val : val * negative_slope;
}

// Bias add kernel: bias shape [C,1,1] broadcast over N,H,W.
// Supports arbitrary input strides; output is contiguous.
__global__ void add_bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ bias,
    int n, int c, int h, int w,
    int in_n_stride, int in_c_stride, int in_h_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * h * w;
    if (idx >= total) return;

    int rem = idx;
    int n_idx = rem / (c * h * w);
    rem -= n_idx * (c * h * w);
    int c_idx = rem / (h * w);
    rem -= c_idx * (h * w);
    int h_idx = rem / w;
    int w_idx = rem - h_idx * w;

    int in_offset = n_idx * in_n_stride + c_idx * in_c_stride + h_idx * in_h_stride + w_idx;
    output[idx] = input[in_offset] + bias[c_idx];
}

// Host launcher for leaky_relu. No PyTorch dependencies.
extern "C" void leaky_relu_launcher(
    float* output,
    const float* input,
    int n, int c, int h, int w,
    int in_n_stride, int in_c_stride, int in_h_stride,
    float negative_slope,
    cudaStream_t stream)
{
    int total = n * c * h * w;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads, 0, stream>>>(
        output, input, n, c, h, w,
        in_n_stride, in_c_stride, in_h_stride,
        negative_slope);
}

// Host launcher for add_bias. No PyTorch dependencies.
extern "C" void add_bias_launcher(
    float* output,
    const float* input,
    const float* bias,
    int n, int c, int h, int w,
    int in_n_stride, int in_c_stride, int in_h_stride,
    cudaStream_t stream)
{
    int total = n * c * h * w;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(
        output, input, bias, n, c, h, w,
        in_n_stride, in_c_stride, in_h_stride);
}
