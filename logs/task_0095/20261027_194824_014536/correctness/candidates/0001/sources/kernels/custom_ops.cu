#include <cuda_runtime.h>

// Non-convolution CUDA kernels for the model:
//   linspace_1d: generate 1D evenly spaced float32 values
//   elementwise_sub: x - y broadcasting over last dimension
// Conv/ConvTranspose are handled in a separate C++ binding using cuDNN,
// not in this .cu file (per contract).

// Kernel: linspace_1d
// Generates steps elements from start to end inclusive.
// Each thread computes at most one element; grid-stride loop handles arbitrary size.
__global__ void linspace_1d_kernel(float* output, int steps, float start, float end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float range = end - start;
    float divisor = (steps > 1) ? (float)(steps - 1) : 1.0f;
    for (int i = idx; i < steps; i += stride) {
        float t = (float)i / divisor;
        output[i] = start + t * range;
    }
}

// Kernel: elementwise_sub with broadcasting of y over the last dimension.
// Input x shape: [batch, channels, length] (3D)
// Input y shape: [length]
// Output shape same as x.
// We assume x is contiguous (contract currently provides contiguous x).
// To be robust, we accept strides; output is contiguous with same shape.
__global__ void elementwise_sub_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ out,
    int batch,
    int channels,
    int length,
    int x_batch_stride,
    int x_channel_stride,
    int y_stride) {
    int total = batch * channels * length;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int l = i % length;
        int c = (i / length) % channels;
        int b = i / (length * channels);
        int x_idx = b * x_batch_stride + c * x_channel_stride + l;
        int y_idx = l * y_stride;  // if y is contiguous, y_stride = 1
        out[i] = x[x_idx] - y[y_idx];
    }
}

// Host launcher for linspace_1d
//   output: pointer to allocated float array of size steps
//   start, end: float endpoints
//   steps: number of elements
//   stream: CUDA stream
// Returns cudaError_t (cudaSuccess on success); caller can check.
extern "C" cudaError_t linspace_1d_launcher(
    float* output,
    float start,
    float end,
    int steps,
    cudaStream_t stream) {
    if (steps <= 0) return cudaSuccess;  // nothing to do, empty tensor
    int threads = 256;
    int blocks = (steps + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;  // grid cap; kernel uses grid-stride loop
    linspace_1d_kernel<<<blocks, threads, 0, stream>>>(output, steps, start, end);
    return cudaGetLastError();
}

// Host launcher for elementwise_sub
//   x: input tensor data (float*) with shape [batch, channels, length]
//   y: input tensor data (float*) with shape [length]
//   out: output tensor data (float*) with same shape as x, contiguous
//   batch, channels, length: dimensions
//   x_batch_stride, x_channel_stride: strides of x (in elements)
//   y_stride: stride of y (usually 1)
//   stream: CUDA stream
// Returns cudaError_t.
extern "C" cudaError_t elementwise_sub_launcher(
    const float* x,
    const float* y,
    float* out,
    int batch,
    int channels,
    int length,
    int x_batch_stride,
    int x_channel_stride,
    int y_stride,
    cudaStream_t stream) {
    if (batch <= 0 || channels <= 0 || length <= 0) return cudaSuccess;
    int total = batch * channels * length;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid-stride loop handles rest
    elementwise_sub_kernel<<<blocks, threads, 0, stream>>>(
        x, y, out, batch, channels, length,
        x_batch_stride, x_channel_stride, y_stride);
    return cudaGetLastError();
}
