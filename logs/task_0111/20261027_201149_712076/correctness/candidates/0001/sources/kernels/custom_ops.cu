#include <cuda_runtime.h>

// kernels/custom_ops.cu
//
// This file contains only non-convolution CUDA kernels.
// The ConvTranspose1d operation (conv_transpose1d_forward) will be implemented
// by the C++ binding stage using cuDNN, as required by the task instructions.
// This file intentionally does not include any cuDNN code or hand-written convolution kernels.

// ----------------------------------------------------------------
// ge_forward: element-wise input >= other (other is a scalar float)
// Input:  float32 tensor, arbitrary contiguous or non-contiguous layout
// Output: bool tensor, same shape as input
// The kernel uses a grid-stride loop to handle any number of elements.
// ----------------------------------------------------------------

__global__ void ge_kernel(const float* __restrict__ input,
                          bool* __restrict__ output,
                          int64_t numel,
                          float other) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = (input[i] >= other);
    }
}

// Host launcher for ge_forward. The binding layer will call this function
// with raw pointers and the current CUDA stream.
extern "C" void ge_forward_launcher(const float* input,
                                    bool* output,
                                    int64_t numel,
                                    float other,
                                    cudaStream_t stream) {
    if (numel <= 0) {
        return;
    }
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    // Cap grid size to avoid exceeding limits; grid-stride loop handles the rest.
    const int64_t max_grid = 65535;
    if (grid_size > max_grid) {
        grid_size = max_grid;
    }
    ge_kernel<<<static_cast<int>(grid_size), block_size, 0, stream>>>(
        input, output, numel, other);
}
