#include <cuda_runtime.h>

// Kernel that copies data from input (arbitrary strides) to output (contiguous)
// Each thread handles one linear output index. Grid-stride loop ensures all elements written even if grid is smaller than total elements.
__global__ void copy_strided_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int total_elems,
    int ndims,
    const int64_t* __restrict__ dims,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ output_strides)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elems; i += stride) {
        // Unravel linear output index into multi-dimensional coordinates
        int64_t remainder = i;
        int64_t input_offset = 0;
        // output is contiguous, so output_strides are used, but can be inferred
        for (int d = ndims - 1; d >= 0; --d) {
            int coord = remainder % dims[d];
            remainder /= dims[d];
            input_offset += coord * input_strides[d];
        }
        output[i] = input[input_offset];
    }
}

// Host launcher: copies input tensor to output tensor while honoring input strides.
// output is assumed to be contiguous in memory (as specified by contract).
// ndims is not used inside kernel except for loop bound; we pass ndims as int.
// Note: output_strides not needed if output is guaranteed contiguous, but we pass for completeness.
extern "C" void stack_like_launcher(
    float* output,
    const float* input,
    int total_elems,
    int ndims,
    const int64_t* dims,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream)
{
    // Configure launch parameters for large tensor
    int threads = 256;
    int blocks = (total_elems + threads - 1) / threads;
    // Cap blocks to avoid excessive grid size; grid-stride loop handles the rest
    const int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;

    copy_strided_kernel<<<blocks, threads, 0, stream>>>(
        output, input, total_elems, ndims,
        dims, input_strides, output_strides);
}
