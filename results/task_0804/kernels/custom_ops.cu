// CUDA kernel and launcher for stack_forward: stacks two contiguous tensors along a new first dimension.
// This file must contain only CUDA kernels and launchers using raw pointers, scalars, and cudaStream_t.
// No torch/ATen/c10/pybind dependencies.

#include <cuda_runtime.h>

// Kernel that copies data from two input tensors into two contiguous slices of the output tensor.
// grid-stride loop to handle large data and avoid exceeding grid dimension limits.
__global__ void stack_forward_kernel(const float* __restrict__ x1,
                                     const float* __restrict__ x2,
                                     float* __restrict__ out,
                                     int64_t numel_per_input) {
    int64_t total_numel = 2 * numel_per_input;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_numel; i += stride) {
        if (i < numel_per_input) {
            out[i] = x1[i];
        } else {
            out[i] = x2[i - numel_per_input];
        }
    }
}

// Launcher for stack_forward. Accepts raw pointers and a CUDA stream.
// x1 and x2 are assumed to be valid CUDA device pointers to contiguous float32 data.
// out is a preallocated device pointer with size 2 * numel_per_input.
// The launcher checks for empty input and returns early without launching.
extern "C" void stack_forward_launcher(const float* x1,
                                       const float* x2,
                                       float* out,
                                       int64_t numel_per_input,
                                       cudaStream_t stream) {
    if (numel_per_input == 0) {
        return;
    }

    // Choose block size and grid size. Grid size capped to avoid exceeding max grid dimension.
    const int block_size = 256;
    int64_t total_numel = 2 * numel_per_input;
    // grid size = min(ceil(total_numel / block_size), max_grid_size)
    // Use int64_t for grid computation; cap at 2^31-1 blocks.
    int64_t grid_size = (total_numel + block_size - 1) / block_size;
    const int64_t max_grid_size = 2147483647LL;  // 2^31-1
    if (grid_size > max_grid_size) {
        grid_size = max_grid_size;
    }

    stack_forward_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        x1, x2, out, numel_per_input);

    // Check for kernel launch errors (optional but recommended).
    // The launcher does not have access to Torch error handling, but we can use cudaGetLastError.
    // However, the caller should also check after the launcher returns.
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) { ... }
    // We'll leave a comment; the binding layer may check cudaGetLastError after call.
}
