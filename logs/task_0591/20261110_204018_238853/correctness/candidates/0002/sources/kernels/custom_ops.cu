// CUDA kernels for custom operations.
// This file contains only the GELU elementwise kernel and launcher.
// Linear (GEMM) will be handled by cuBLAS in the binding stage, not here.
// All launchers use raw pointers, int64_t sizes, and cudaStream_t.

#include <cuda_runtime.h>
#include <math_constants.h>

// GELU activation: 0.5 * x * (1 + erf(x / sqrt(2)))
__global__ void gelu_kernel(float* output, const float* input, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        output[i] = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
    }
}

// Launcher for GELU forward.
// Allocates or receives output pointer from caller? We follow pattern from SKILL.md:
// launcher takes output pointer, input pointer, size, stream.
// The binding.cpp will allocate output via torch::empty_like and pass pointers.
extern "C" void gelu_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream) {
    const int block_size = 256;
    int64_t blocks = (numel + block_size - 1) / block_size;
    // Grid-stride loop handles large sizes; grid dimension max is 2^31-1.
    // If numel is extremely large, we could split into multiple launches, but
    // grid-stride within a single launch with capped grid size is enough.
    int max_grid = 65535 * 32; // typical max grid size for CUDA (may vary)
    if (blocks > max_grid) blocks = max_grid;
    gelu_kernel<<<blocks, block_size, 0, stream>>>(output, input, numel);
}
