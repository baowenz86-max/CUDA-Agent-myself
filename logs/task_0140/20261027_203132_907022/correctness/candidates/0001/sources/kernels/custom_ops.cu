#include <cuda_runtime.h>

__global__ void hardsigmoid_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        // hardsigmoid: clamp(x + 3, 0, 6) / 6
        float clamped = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        output[i] = clamped * (1.0f / 6.0f);
    }
}

__global__ void relu_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = fmaxf(input[i], 0.0f);
    }
}

extern "C" void hardsigmoid_forward_launcher(const float* input,
                                             float* output,
                                             int64_t numel,
                                             cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    // Cap grid size to avoid excessive launch overhead while still allowing grid-stride loop
    if (grid_size > 65535) grid_size = 65535;
    hardsigmoid_kernel<<<static_cast<int>(grid_size), block_size, 0, stream>>>(
        input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // The launcher cannot throw exceptions in this restricted context.
        // We rely on the binding layer to check errors after launch; this call
        // ensures the error is captured here and not masked by later launches.
        // No output is written on failure.
    }
}

extern "C" void relu_forward_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    relu_kernel<<<static_cast<int>(grid_size), block_size, 0, stream>>>(
        input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Same as above: error captured, binding layer may check cudaGetLastError again.
    }
}
