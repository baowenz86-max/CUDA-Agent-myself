#include <cuda_runtime.h>
#include <cstdint>

// Element-wise absolute value kernel
__global__ void custom_abs_kernel(float* output, const float* input, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

// Element-wise tangent kernel
__global__ void custom_tan_kernel(float* output, const float* input, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = tanf(input[i]);
    }
}

// Launcher for custom_abs
extern "C" void custom_abs_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream) {
    const int block_size = 256;
    int64_t blocks_needed = (numel + block_size - 1) / block_size;
    int grid_size = (blocks_needed > 65535) ? 65535 : static_cast<int>(blocks_needed);
    if (grid_size < 1) grid_size = 1;
    custom_abs_kernel<<<grid_size, block_size, 0, stream>>>(output, input, numel);
}

// Launcher for custom_tan
extern "C" void custom_tan_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream) {
    const int block_size = 256;
    int64_t blocks_needed = (numel + block_size - 1) / block_size;
    int grid_size = (blocks_needed > 65535) ? 65535 : static_cast<int>(blocks_needed);
    if (grid_size < 1) grid_size = 1;
    custom_tan_kernel<<<grid_size, block_size, 0, stream>>>(output, input, numel);
}
