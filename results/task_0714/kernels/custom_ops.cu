#include <cuda_runtime.h>
#include <math.h>

// Elementwise absolute value: y = |x|
__global__ void abs_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           int64_t numel) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

// Elementwise softplus with beta=1, threshold=20
__global__ void softplus_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t numel) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        output[i] = (x > 20.0f) ? x : log1pf(expf(x));
    }
}

// Launcher for abs_forward
extern "C" void abs_forward_launcher(const float* input,
                                     float* output,
                                     int64_t numel,
                                     cudaStream_t stream) {
    const int block_size = 256;
    int64_t num_blocks = (numel + block_size - 1) / block_size;
    if (num_blocks > 0) {
        // Limit grid size to avoid exceeding device limits
        int max_blocks = 65535;
        int grid = (int)min((int64_t)max_blocks, num_blocks);
        abs_kernel<<<grid, block_size, 0, stream>>>(input, output, numel);
    }
}

// Launcher for softplus_forward
extern "C" void softplus_forward_launcher(const float* input,
                                          float* output,
                                          int64_t numel,
                                          cudaStream_t stream) {
    const int block_size = 256;
    int64_t num_blocks = (numel + block_size - 1) / block_size;
    if (num_blocks > 0) {
        int max_blocks = 65535;
        int grid = (int)min((int64_t)max_blocks, num_blocks);
        softplus_kernel<<<grid, block_size, 0, stream>>>(input, output, numel);
    }
}
