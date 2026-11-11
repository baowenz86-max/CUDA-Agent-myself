// CUDA kernel for fused abs + flatten
// Input: float32 tensor of shape [1280, 3, 512, 256]
// Output: contiguous float32 tensor of shape [1280, 393216]
// The flatten operation is a pure reshape: for contiguous input, the output
// linear layout is the same as input linear layout. Since torch.abs
// preserves memory layout and input is contiguous, we simply compute
// absolute value for every element in linear order.
#include <cuda_runtime.h>
#include <stdint.h>

// Kernel: each thread computes fabsf for elements indexed by global thread id
__global__ void abs_flat_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t numel) {
    // grid-stride loop
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < numel;
         idx += (int64_t)gridDim.x * blockDim.x) {
        output[idx] = fabsf(input[idx]);
    }
}

// Host launcher: input and output are already allocated CUDA memory.
// numel is the total number of elements (input.numel() == output.numel()).
extern "C" void custom_abs_flat_forward_launcher(const float* input,
                                                  float* output,
                                                  int64_t numel,
                                                  cudaStream_t stream) {
    // Use 256 threads per block, typical for this kind of memory-bound kernel
    const int block_size = 256;
    // Calculate number of blocks needed, with a cap to avoid excessive blocks
    int64_t blocks = (numel + block_size - 1) / block_size;
    // Limit grid size to avoid kernel launch failure for very large numel
    // 2^31-1 blocks is the maximum for 1D grid; but we don't need that many,
    // use at most 2^20 blocks (约1M blocks) to ensure grid-stride loop works.
    if (blocks > (1 << 20)) blocks = 1 << 20;
    
    abs_flat_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
        input, output, numel);
}
