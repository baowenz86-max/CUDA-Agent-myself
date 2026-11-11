// kernels/custom_ops.cu
// This file contains CUDA kernels and launchers for custom operations.
// It is restricted to raw CUDA kernels and launchers using primitive types
// and cudaStream_t. No PyTorch/ATen/pybind dependencies are allowed here.
//
// For conv3d_forward, cuDNN is required and will be implemented in a separate
// C++ file (third stage). This .cu file does not implement conv3d_forward.

#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// tanh_forward: element-wise hyperbolic tangent
// -----------------------------------------------------------------------------

// Kernel: computes tanh for each element.
// Uses grid-stride loop to handle arbitrary number of elements.
__global__ void tanh_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = tanhf(input[i]);
    }
}

// Launcher: function to be called from binding.cpp
extern "C" void tanh_forward_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      cudaStream_t stream) {
    if (numel <= 0) {
        return;
    }
    const int block_size = 256;
    // Calculate grid size, capped to avoid excessive blocks.
    int64_t grid_size = (numel + block_size - 1) / block_size;
    // Limit grid size to avoid too many blocks (e.g., up to 2^31-1, but we cap
    // to a reasonable maximum based on device, or use a fixed maximum).
    // Here we use a maximum of 65535 blocks in x dimension for compatibility;
    // grid-stride loop handles larger sizes.
    const int64_t max_grid_x = 65535;
    if (grid_size > max_grid_x) {
        grid_size = max_grid_x;
    }
    tanh_kernel<<<(int)grid_size, block_size, 0, stream>>>(input, output, numel);
}

// Note: conv3d_forward is not implemented here because it requires cuDNN.
// The binding for conv3d_forward will be provided by a separate C++ file
// that uses cuDNN. This .cu file only contains the tanh kernel.
