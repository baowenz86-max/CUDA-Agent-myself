#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel for LeakyReLU forward: out[i] = in[i] >= 0 ? in[i] : negative_slope * in[i]
__global__ void leaky_relu_kernel(float* __restrict__ out,
                                 const float* __restrict__ in,
                                 int64_t n,
                                 float negative_slope) {
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < n; i += stride) {
        float v = in[i];
        out[i] = (v >= 0.0f) ? v : (negative_slope * v);
    }
}

// Launcher with C linkage for binding layer
extern "C" void custom_leaky_relu_launcher(float* out,
                                           const float* in,
                                           int64_t n,
                                           double negative_slope,
                                           cudaStream_t stream) {
    if (n == 0) return;

    const int block_size = 256;
    // Calculate grid size, cap to maximum grid dimension (2^31-1) per block dim, but use grid-stride loop anyway
    int64_t blocks64 = (n + block_size - 1) / block_size;
    int grid_size = (int)((blocks64 > 2147483647LL) ? 2147483647 : blocks64);
    if (grid_size < 1) grid_size = 1;

    float slope = (float)negative_slope;
    leaky_relu_kernel<<<grid_size, block_size, 0, stream>>>(out, in, n, slope);
}
