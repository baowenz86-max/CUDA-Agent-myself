// kernels/custom_ops.cu
// CUDA kernel for element-wise absolute value.
// This file contains only the CUDA kernel and a C-style launcher.
// The launcher takes raw pointers, numel, and a CUDA stream.
// Binding layer (kernels/custom_abs_binding.cpp) will handle PyTorch tensor
// allocation, validation, and stream retrieval.

#include <cuda_runtime.h>

// Kernel: each thread computes absolute value for multiple elements via grid-stride loop.
// Supports arbitrary large sizes (numel can exceed grid dimension).
// Memory access is coalesced when the input is contiguous; for non-contiguous strides,
// the caller is expected to pass a contiguous copy if needed. However, the launcher
// only receives raw pointer; binding layer should ensure input is contiguous or provide
// a contiguous copy. According to contract, output should be contiguous and shape same.
// Since torch.abs on a contiguous input returns a contiguous tensor with same strides.
// Therefore, in binding layer we can assume contiguous input and output.
__global__ void abs_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = fabsf(val);
    }
}

// Launcher for custom_abs.
// Parameters:
//   input: pointer to device memory containing input tensor (float32)
//   output: pointer to device memory where output will be written (float32)
//   numel: total number of elements in tensor
//   stream: CUDA stream to execute kernel on
// The caller must ensure output has enough memory for numel floats.
// No error checking is done here; the binding layer should call cudaGetLastError.
// The kernel is launched with 256 threads per block; number of blocks is chosen
// to cover all elements, but the grid-stride loop handles any size.
extern "C" void custom_abs_launcher(const float* input, float* output,
                                    int64_t numel, cudaStream_t stream) {
    const int threads_per_block = 256;
    // Calculate number of blocks: at least 1, and enough to cover numel.
    // But grid-stride loop allows fewer blocks; we use up to a maximum block count
    // to avoid excessive grid dimension. Use min needed to cover numel with one pass,
    // but not exceeding MAX_GRID_DIM.
    int blocks = (int)((numel + threads_per_block - 1) / threads_per_block);
    // Optionally could cap blocks to e.g. 65535 * 32 for large sizes, but grid-stride loop handles
    // even if blocks is 0 (but we ensure at least 1).
    if (blocks < 1) blocks = 1;
    // Note: If numel is extremely large, blocks could exceed maximum grid dimension in x (2^31-1).
    // In that case, we use the maximum grid size and rely on loop iterations.
    const int max_grid_x = 2147483647; // 2^31 - 1
    if (blocks > max_grid_x) {
        blocks = max_grid_x;
    }

    // Launch kernel
    abs_kernel<<<blocks, threads_per_block, 0, stream>>>(input, output, numel);

    // Optional: Check for launch errors (can be enabled if needed)
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) { ... }
}
