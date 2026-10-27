#include <cuda_runtime.h>

// Elementwise broadcast multiply: output[b,c,l] = x[b,c,l] * bias[c]
// x: shape (batch_size, out_channels, length_out), bias: shape (out_channels, 1) broadcast.
// This kernel assumes contiguous inputs (binding should ensure contiguity).
__global__ void custom_mul_kernel(const float* __restrict__ x,
                                  const float* __restrict__ bias,
                                  float* __restrict__ output,
                                  int64_t batch_size,
                                  int64_t out_channels,
                                  int64_t length_out) {
    int64_t total = batch_size * out_channels * length_out;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int64_t c = (idx / length_out) % out_channels;
        output[idx] = x[idx] * bias[c];
    }
}

// Launcher for custom_mul, to be called from the C++ binding.
extern "C" void custom_mul_launcher(const float* x,
                                    const float* bias,
                                    float* output,
                                    int64_t batch_size,
                                    int64_t out_channels,
                                    int64_t length_out,
                                    cudaStream_t stream) {
    int64_t total = batch_size * out_channels * length_out;
    const int threads = 256;
    // Grid-stride loop handles arbitrary sizes; use enough blocks to cover all elements.
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (blocks > 65535) {
        blocks = 65535;
    }
    custom_mul_kernel<<<blocks, threads, 0, stream>>>(x, bias, output,
                                                      batch_size, out_channels, length_out);
}
