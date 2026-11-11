#include <cuda_runtime.h>
#include <math.h>

// Elementwise hyperbolic tangent kernel for float32 tensors.
// Grid-stride loop ensures all elements are processed even if grid size
// does not cover the entire tensor.
__global__ void custom_tanh_kernel(float* __restrict__ output,
                                   const float* __restrict__ input,
                                   int64_t numel) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = tanhf(input[i]);
    }
}

// C-interface launcher for the custom tanh operation.
// The binding layer (C++) will call this function after validating
// that input and output are CUDA tensors of dtype float32 and that
// output is allocated with the same shape as input.
// No PyTorch dependencies are included here.
extern "C" void custom_tanh_launcher(float* output,
                                     const float* input,
                                     int64_t numel,
                                     cudaStream_t stream) {
    // Launch kernel with 256 threads per block and enough blocks to
    // saturate the GPU, but with grid-stride loop the grid can be limited.
    // Use grid size capped at 32 blocks per SM times number of SMs? Simpler:
    // blockDim = 256, gridDim = min((numel + 255) / 256, 4096) which is ample.
    int block_size = 256;
    int64_t num_blocks_64 = (numel + block_size - 1) / block_size;
    if (num_blocks_64 > 4096) num_blocks_64 = 4096;
    int num_blocks = (int)num_blocks_64;
    custom_tanh_kernel<<<num_blocks, block_size, 0, stream>>>(
        output, input, numel);
}
