#include <cuda_runtime.h>
#include <cmath>

// Elementwise hard shrink: y = x if abs(x) > 0.5 else 0
__global__ void hardshrink_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int64_t total) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        float x = input[i];
        output[i] = (fabsf(x) > 0.5f) ? x : 0.0f;
    }
}

// Elementwise ELU with alpha=1.0: y = x if x >= 0 else exp(x)-1
__global__ void elu_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           int64_t total) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        float x = input[i];
        output[i] = (x >= 0.0f) ? x : (expf(x) - 1.0f);
    }
}

// Fused hardshrink then ELU, single pass
__global__ void fused_hardshrink_elu_kernel(const float* __restrict__ input,
                                            float* __restrict__ output,
                                            int64_t total) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        float x = input[i];
        // hardshrink
        float h = (fabsf(x) > 0.5f) ? x : 0.0f;
        // elu
        output[i] = (h >= 0.0f) ? h : (expf(h) - 1.0f);
    }
}

// Helper to compute grid size (blocks) for given total elements and block size
static inline int64_t compute_grid(int64_t total, int block_size) {
    if (total <= 0) return 0;
    return (total + block_size - 1) / block_size;
}

extern "C" void hardshrink_forward_launcher(const float* input,
                                            float* output,
                                            int64_t total,
                                            cudaStream_t stream) {
    if (total == 0) return;
    const int block_size = 256;
    int64_t grid = compute_grid(total, block_size);
    hardshrink_kernel<<<grid, block_size, 0, stream>>>(input, output, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // The binding layer will check errors; we don't throw here.
    }
}

extern "C" void elu_forward_launcher(const float* input,
                                     float* output,
                                     int64_t total,
                                     cudaStream_t stream) {
    if (total == 0) return;
    const int block_size = 256;
    int64_t grid = compute_grid(total, block_size);
    elu_kernel<<<grid, block_size, 0, stream>>>(input, output, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore here, binding will check
    }
}

extern "C" void hardshrink_elu_forward_launcher(const float* input,
                                                float* output,
                                                int64_t total,
                                                cudaStream_t stream) {
    if (total == 0) return;
    const int block_size = 256;
    int64_t grid = compute_grid(total, block_size);
    fused_hardshrink_elu_kernel<<<grid, block_size, 0, stream>>>(input, output, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore here, binding will check
    }
}
