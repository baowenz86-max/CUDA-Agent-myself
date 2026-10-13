#include <cuda_runtime.h>

__global__ void sigmoid_forward_kernel(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = 1.0f / (1.0f + expf(-val));
    }
}

extern "C" void sigmoid_forward_launcher(const float* input,
                                         float* output,
                                         int64_t numel,
                                         cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 2147483647) {
        grid_size = 2147483647;
    }
    sigmoid_forward_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input, output, numel);
}
