#include <cuda_runtime.h>

// Elementwise Hardswish: y = x * min(max(x + 3, 0), 6) / 6
__global__ void hardswish_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        output[i] = x * relu6 * (1.0f / 6.0f);
    }
}

// Elementwise Softplus: y = log(1 + exp(x)) for x <= 20, else x
__global__ void softplus_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        output[i] = (x > 20.0f) ? x : log1pf(expf(x));
    }
}

// Launcher for hardswish elementwise op
void hardswish_forward_launcher(const float* input,
                                float* output,
                                int64_t numel,
                                cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    // Cap grid size to avoid exceeding maximum; grid-stride loop handles large sizes
    if (grid_size > 65535) grid_size = 65535;
    hardswish_kernel<<<(int)grid_size, block_size, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real binding, raise a Python exception; here we simply leave it for the
        // caller to check via cudaGetLastError after synchronizing.
    }
}

// Launcher for softplus elementwise op
void softplus_forward_launcher(const float* input,
                               float* output,
                               int64_t numel,
                               cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    softplus_kernel<<<(int)grid_size, block_size, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // error handling left to binding layer
    }
}
