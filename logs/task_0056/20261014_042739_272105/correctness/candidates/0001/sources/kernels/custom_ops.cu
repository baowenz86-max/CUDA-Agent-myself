#include <cuda_runtime.h>
#include <cmath>

// Sigmoid forward kernel: y = 1 / (1 + exp(-x))
// Supports arbitrary strides via 4D indexing.
__global__ void sigmoid_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int stride_n, int stride_c, int stride_h, int stride_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    // Decompose linear index into coordinates
    int n = idx / (C * H * W);
    int rem = idx % (C * H * W);
    int c = rem / (H * W);
    rem = rem % (H * W);
    int h = rem / W;
    int w = rem % W;

    int offset = n * stride_n + c * stride_c + h * stride_h + w * stride_w;
    float val = x[offset];
    y[offset] = 1.0f / (1.0f + expf(-val));
}

// BatchNorm2d inference kernel: y = (x - mean[c]) / sqrt(var[c] + eps) * weight[c] + bias[c]
// Supports arbitrary strides via 4D indexing.
__global__ void batch_norm2d_inference_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int stride_n, int stride_c, int stride_h, int stride_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int n = idx / (C * H * W);
    int rem = idx % (C * H * W);
    int c = rem / (H * W);
    rem = rem % (H * W);
    int h = rem / W;
    int w = rem % W;

    int offset = n * stride_n + c * stride_c + h * stride_h + w * stride_w;
    float mean = running_mean[c];
    float var = running_var[c];
    float scale = weight[c] / sqrtf(var + eps);
    float shift = bias[c] - mean * scale;
    y[offset] = x[offset] * scale + shift;
}

// Launcher for sigmoid_forward
// x: input tensor data pointer, y: output tensor data pointer
// N,C,H,W: dimensions
// stride_n, stride_c, stride_h, stride_w: strides in elements
// stream: CUDA stream
extern "C" void sigmoid_forward_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    int stride_n, int stride_c, int stride_h, int stride_w,
    cudaStream_t stream) {
    int total = N * C * H * W;
    if (total <= 0) return;

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    // Optional: cap grid size to avoid too many blocks, use grid-stride loop if needed.
    // For simplicity here, we launch exactly enough blocks; current maximum total is ~25M, grid size ~98K, okay.
    sigmoid_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        x, y, N, C, H, W, stride_n, stride_c, stride_h, stride_w);
}

// Launcher for batch_norm2d_inference
// x, weight, bias, running_mean, running_var: input tensors (float32)
// eps: epsilon
// y: output tensor
// N,C,H,W: dimensions
// stride_n, stride_c, stride_h, stride_w: strides of x in elements
// stream: CUDA stream
extern "C" void batch_norm2d_inference_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    float* y,
    int N, int C, int H, int W,
    int stride_n, int stride_c, int stride_h, int stride_w,
    cudaStream_t stream) {
    int total = N * C * H * W;
    if (total <= 0) return;

    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    batch_norm2d_inference_kernel<<<grid_size, block_size, 0, stream>>>(
        x, weight, bias, running_mean, running_var, eps, y,
        N, C, H, W, stride_n, stride_c, stride_h, stride_w);
}
