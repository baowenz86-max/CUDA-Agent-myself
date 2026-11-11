#include <cuda_runtime.h>

// relu_add_div6: y = max(x + 3, 0) / 6
__global__ void relu_add_div6_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i] + 3.0f;
        val = val > 0.0f ? val : 0.0f;
        output[i] = val * (1.0f / 6.0f);
    }
}

// hardswish: y = x * min(max(x + 3, 0), 6) / 6
__global__ void hardswish_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i] + 3.0f;
        val = val > 0.0f ? val : 0.0f;
        val = val < 6.0f ? val : 6.0f;
        output[i] = input[i] * val * (1.0f / 6.0f);
    }
}

extern "C" void relu_add_div6_launcher(float* output,
                                        const float* input,
                                        int64_t numel,
                                        cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) {
        grid_size = 65535;
    }
    relu_add_div6_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error will be checked by binding after launch; we could print but avoid dependencies.
    }
}

extern "C" void hardswish_launcher(float* output,
                                    const float* input,
                                    int64_t numel,
                                    cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) {
        grid_size = 65535;
    }
    hardswish_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error will be checked by binding after launch; we could print but avoid dependencies.
    }
}
