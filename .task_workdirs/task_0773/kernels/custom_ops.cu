#include <cuda_runtime.h>
#include <stdint.h>

__global__ void stack_dim0_kernel(const float* x, const float* y, float* output,
                                  int64_t outer_size, int64_t inner_size) {
    int64_t total_output = outer_size * 2 * inner_size;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < total_output; i += stride) {
        int64_t suffix = i % inner_size;
        int64_t tmp = i / inner_size;
        int64_t j = tmp % 2;
        int64_t prefix = tmp / 2;
        int64_t src_idx = prefix * inner_size + suffix;
        if (j == 0) {
            output[i] = x[src_idx];
        } else {
            output[i] = y[src_idx];
        }
    }
}

extern "C" void stack_dim0_launcher(const float* x, const float* y, float* output,
                                    int64_t outer_size, int64_t inner_size, cudaStream_t stream) {
    int64_t total_output = outer_size * 2 * inner_size;
    int block = 256;
    int64_t num_blocks = (total_output + block - 1) / block;
    if (num_blocks > 2147483647) num_blocks = 2147483647;
    int grid = (int)num_blocks;
    stack_dim0_kernel<<<grid, block, 0, stream>>>(x, y, output, outer_size, inner_size);
}

__global__ void all_dim0_kernel(const float* input, bool* output, int64_t inner_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < inner_size; i += stride) {
        float v0 = input[i];
        float v1 = input[i + inner_size];
        output[i] = (v0 != 0.0f) && (v1 != 0.0f);
    }
}

extern "C" void all_dim0_launcher(const float* input, bool* output, int64_t inner_size, cudaStream_t stream) {
    int block = 256;
    int64_t num_blocks = (inner_size + block - 1) / block;
    if (num_blocks > 2147483647) num_blocks = 2147483647;
    int grid = (int)num_blocks;
    all_dim0_kernel<<<grid, block, 0, stream>>>(input, output, inner_size);
}
