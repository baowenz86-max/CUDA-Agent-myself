#include <cuda_runtime.h>

__global__ void relu_forward_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

__global__ void abs_forward_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

extern "C" void relu_forward_launcher(float* output,
                                      const float* input,
                                      int64_t numel,
                                      cudaStream_t stream) {
    const int64_t block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) {
        grid_size = 65535;
    }
    relu_forward_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel);
}

extern "C" void abs_forward_launcher(float* output,
                                     const float* input,
                                     int64_t numel,
                                     cudaStream_t stream) {
    const int64_t block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) {
        grid_size = 65535;
    }
    abs_forward_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel);
}
