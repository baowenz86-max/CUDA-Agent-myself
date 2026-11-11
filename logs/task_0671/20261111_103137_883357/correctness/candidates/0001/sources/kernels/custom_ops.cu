#include <cuda_runtime.h>

// Element-wise absolute value: y = |x|
__global__ void abs_kernel(const float* __restrict__ input,
                          float* __restrict__ output,
                          int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

// Element-wise floor: y = floor(x)
__global__ void floor_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        output[i] = floorf(input[i]);
    }
}

// Launcher for custom_abs
extern "C" void custom_abs_launcher(const float* input,
                                     float* output,
                                     int64_t n,
                                     cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (n + block_size - 1) / block_size;
    // Cap grid size to avoid exceeding max grid dimension (2^31-1)
    const int64_t max_grid = 2147483647LL;
    if (grid_size > max_grid) grid_size = max_grid;
    abs_kernel<<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(input, output, n);
}

// Launcher for custom_floor
extern "C" void custom_floor_launcher(const float* input,
                                      float* output,
                                      int64_t n,
                                      cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (n + block_size - 1) / block_size;
    const int64_t max_grid = 2147483647LL;
    if (grid_size > max_grid) grid_size = max_grid;
    floor_kernel<<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(input, output, n);
}
