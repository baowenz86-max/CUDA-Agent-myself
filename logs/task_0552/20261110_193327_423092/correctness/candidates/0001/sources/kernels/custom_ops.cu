#include <cuda_runtime.h>

// ELU activation: y = x > 0 ? x : exp(x) - 1
__global__ void elu_kernel_forward(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        float v = x[idx];
        y[idx] = (v > 0.0f) ? v : (expf(v) - 1.0f);
    }
}

// Absolute value: y = fabsf(x)
__global__ void abs_kernel_forward(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        y[idx] = fabsf(x[idx]);
    }
}

// Fused abs(ELU(x)): y = x > 0 ? x : 1 - exp(x)
__global__ void elu_abs_kernel_forward(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        float v = x[idx];
        y[idx] = (v > 0.0f) ? v : (1.0f - expf(v));
    }
}

extern "C" void elu_forward_launcher(const float* x, float* y, int64_t n, cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (n + block_size - 1) / block_size;
    // 限制 grid 大小避免超出硬件限制（虽然通常不会，但安全）
    if (grid_size > 2147483647LL) grid_size = 2147483647LL;
    elu_kernel_forward<<<(unsigned int)grid_size, block_size, 0, stream>>>(x, y, n);
}

extern "C" void abs_forward_launcher(const float* x, float* y, int64_t n, cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (n + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL;
    abs_kernel_forward<<<(unsigned int)grid_size, block_size, 0, stream>>>(x, y, n);
}

extern "C" void elu_abs_forward_launcher(const float* x, float* y, int64_t n, cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (n + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL;
    elu_abs_kernel_forward<<<(unsigned int)grid_size, block_size, 0, stream>>>(x, y, n);
}
