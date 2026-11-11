// kernels/custom_ops.cu
// All CUDA kernels and launchers for non-convolution ops.
// Conv2d is implemented with cuDNN in the binding .cpp file.

#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// Kernel: batch_norm2d_inference
// y[n,c,h,w] = (x[n,c,h,w] - mean[c]) / sqrt(var[c] + eps) * weight[c] + bias[c]
// weight and bias may be NULL (treated as 1 and 0)
// -----------------------------------------------------------------------------
__global__ void batch_norm2d_inference_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ mean,
    const float* __restrict__ var,
    float eps,
    int C,
    int HW,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        int c = (i / HW) % C;  // channel index
        float val = x[i];
        float m = mean[c];
        float v = var[c] + eps;
        float scale = (weight != nullptr) ? weight[c] : 1.0f;
        float shift = (bias != nullptr) ? bias[c] : 0.0f;
        y[i] = (val - m) * rsqrtf(v) * scale + shift;
    }
}

// Launcher for batch_norm2d_inference
extern "C" void batch_norm2d_inference_launcher(
    float* y,
    const float* x,
    const float* weight,
    const float* bias,
    const float* mean,
    const float* var,
    float eps,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int64_t HW = (int64_t)H * W;
    int64_t total = (int64_t)N * C * HW;
    int threads = 256;
    int blocks = (int)min((int64_t)65535, (total + threads - 1) / threads);
    batch_norm2d_inference_kernel<<<blocks, threads, 0, stream>>>(
        y, x, weight, bias, mean, var, eps, C, (int)HW, total);
}

// -----------------------------------------------------------------------------
// Kernel: leaky_relu
// y[i] = x[i] >= 0 ? x[i] : negative_slope * x[i]
// -----------------------------------------------------------------------------
__global__ void leaky_relu_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    float negative_slope,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        float val = x[i];
        y[i] = val >= 0.0f ? val : negative_slope * val;
    }
}

// Launcher for leaky_relu
extern "C" void leaky_relu_launcher(
    float* y,
    const float* x,
    float negative_slope,
    int64_t total_elements,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)min((int64_t)65535, (total_elements + threads - 1) / threads);
    leaky_relu_kernel<<<blocks, threads, 0, stream>>>(
        y, x, negative_slope, total_elements);
}

// -----------------------------------------------------------------------------
// Kernel: add_bias_broadcast
// y[n,c,h,w] = x[n,c,h,w] + bias[c,0,0] where bias has shape [C,1,1]
// Input x assumed contiguous; strides passed from binding.
// -----------------------------------------------------------------------------
__global__ void add_bias_broadcast_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ bias,
    int C,
    int HW,
    int64_t total_elements,
    int64_t x_stride_n,   // stride in elements for N dimension
    int64_t x_stride_c,   // stride in elements for C dimension
    int64_t x_stride_hw)  // stride in elements for HW dimension (usually 1)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        int n = i / (C * HW);
        int rem = i % (C * HW);
        int c = rem / HW;
        int hw = rem % HW;
        int64_t x_idx = n * x_stride_n + c * x_stride_c + hw * x_stride_hw;
        y[i] = x[x_idx] + bias[c];
    }
}

// Launcher for add_bias_broadcast
extern "C" void add_bias_broadcast_launcher(
    float* y,
    const float* x,
    const float* bias,
    int N, int C, int H, int W,
    int64_t x_stride_n,
    int64_t x_stride_c,
    int64_t x_stride_hw,
    cudaStream_t stream)
{
    int HW = H * W;
    int64_t total = (int64_t)N * C * HW;
    int threads = 256;
    int blocks = (int)min((int64_t)65535, (total + threads - 1) / threads);
    add_bias_broadcast_kernel<<<blocks, threads, 0, stream>>>(
        y, x, bias, C, HW, total, x_stride_n, x_stride_c, x_stride_hw);
}
