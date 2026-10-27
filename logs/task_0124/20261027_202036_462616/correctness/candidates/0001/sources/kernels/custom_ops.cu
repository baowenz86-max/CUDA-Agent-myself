#include <cuda_runtime.h>

// Elementwise clamp kernel: out[i] = min(max(in[i], min_val), max_val)
__global__ void clamp_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             int64_t numel,
                             float min_val,
                             float max_val) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        val = val < min_val ? min_val : val;
        val = val > max_val ? max_val : val;
        output[i] = val;
    }
}

// Elementwise power kernel: out[i] = powf(in[i], exponent)
__global__ void pow_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           int64_t numel,
                           float exponent) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = powf(input[i], exponent);
    }
}

// Launcher for clamp_forward
extern "C" void clamp_forward_launcher(float* output,
                                       const float* input,
                                       int64_t numel,
                                       float min_val,
                                       float max_val,
                                       cudaStream_t stream) {
    if (numel == 0) return;
    const int block_size = 256;
    // 限制 grid 大小以避免超过硬件限制，kernel 内部使用 grid-stride loop
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    clamp_kernel<<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(
        input, output, numel, min_val, max_val);
}

// Launcher for pow_forward
extern "C" void pow_forward_launcher(float* output,
                                      const float* input,
                                      int64_t numel,
                                      float exponent,
                                      cudaStream_t stream) {
    if (numel == 0) return;
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    pow_kernel<<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(
        input, output, numel, exponent);
}
