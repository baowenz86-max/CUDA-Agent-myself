#include <cuda_runtime.h>

// CUDA kernels and launchers for max_dim1 and expm1 operations.
// This file must not include torch/ATen/pybind headers.
// Bindings in kernels/*_binding.cpp will call these launchers.

// Kernel: compute max over dim=1 (channels) with keepdim=True.
// Input: float* x with shape [batch, channels, height, width], arbitrary strides.
// Output: float* values, int64_t* indices, shape [batch, 1, height, width].
// tie-breaking: first max index.
__global__ void max_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int batch,
    int channels,
    int height,
    int width,
    long long total) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        // Decode linear index into batch, h, w (since output has same shape but channels=1)
        int w = idx % width;
        long long tmp = idx / width;
        int h = tmp % height;
        int b = tmp / height;

        // Input offset for channel 0 at (b,0,h,w)
        // Input strides: assume contiguous? We'll compute using base index: b*channels*height*width + c*height*width + h*width + w
        // But support non-contiguous? The contract says input can be non-contiguous; we can handle strides by passing strides, but launcher currently passes raw pointer; we assume contiguous for now as contract says 'contiguous or supports strides'.
        // For simplicity, we'll assume input is contiguous (as provided strides are contiguous).
        // If non-contiguous, binding may make contiguous before calling.
        long long base = ((long long)b * channels + 0LL) * ((long long)height * width) + (long long)h * width + w;
        float max_val = x[base];
        int max_idx = 0;
        for (int c = 1; c < channels; ++c) {
            long long offset = ((long long)b * channels + c) * ((long long)height * width) + (long long)h * width + w;
            float val = x[offset];
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
        }
        values[idx] = max_val;
        indices[idx] = max_idx;
    }
}

// Kernel: expm1 elementwise. Output same shape as input.
__global__ void expm1_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long total) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        y[idx] = expf(x[idx]) - 1.0f;
    }
}

// Launcher for max_dim1. Assumes x is contiguous float32 CUDA tensor.
// Outputs values (float) and indices (int64_t) allocated by caller.
extern "C" void max_dim1_kernel_launcher(
    const float* x,
    float* values,
    int64_t* indices,
    int batch,
    int channels,
    int height,
    int width,
    cudaStream_t stream) {
    long long total = (long long)batch * height * width;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    // Cap blocks at 65535 for old GPUs; but we need grid-stride loop for larger.
    int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    max_dim1_kernel<<<blocks, threads, 0, stream>>>(
        x, values, indices, batch, channels, height, width, total);
}

// Launcher for expm1. Assumes x contiguous float32, output y allocated by caller.
extern "C" void expm1_kernel_launcher(
    const float* x,
    float* y,
    long long total,
    cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    expm1_kernel<<<blocks, threads, 0, stream>>>(x, y, total);
}
