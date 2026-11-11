#include <cuda_runtime.h>

// Elementwise absolute value
__global__ void custom_abs_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int64_t total_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

// Cumulative sum along dimension 0.
// Each thread handles one index in the flattened (dim1*dim2*...*dimN) space.
// It loops over dim0, accumulating values from input at position [d0, inner_idx].
__global__ void custom_cumsum_dim0_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int64_t dim0,
                                           int64_t inner_size) {
    int64_t inner_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    if (inner_idx >= inner_size) return;

    for (int64_t idx = inner_idx; idx < inner_size; idx += stride) {
        float acc = 0.0f;
        for (int64_t d0 = 0; d0 < dim0; ++d0) {
            acc += input[d0 * inner_size + idx];
            output[d0 * inner_size + idx] = acc;
        }
    }
}

// Launcher for custom_abs
extern "C" void custom_abs_launcher(const float* input,
                                     float* output,
                                     int64_t total_elements,
                                     cudaStream_t stream) {
    int threads = 256;
    int blocks = static_cast<int>((total_elements + threads - 1) / threads);
    const int max_blocks_per_grid = 65535;
    if (blocks > max_blocks_per_grid) {
        blocks = max_blocks_per_grid;
    }
    custom_abs_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements);
}

// Launcher for custom_cumsum along dim=0
extern "C" void custom_cumsum_launcher(const float* input,
                                         float* output,
                                         int64_t dim0,
                                         int64_t inner_size,
                                         cudaStream_t stream) {
    int threads = 256;
    int blocks = static_cast<int>((inner_size + threads - 1) / threads);
    const int max_blocks_per_grid = 65535;
    if (blocks > max_blocks_per_grid) {
        blocks = max_blocks_per_grid;
    }
    custom_cumsum_dim0_kernel<<<blocks, threads, 0, stream>>>(input, output, dim0, inner_size);
}
