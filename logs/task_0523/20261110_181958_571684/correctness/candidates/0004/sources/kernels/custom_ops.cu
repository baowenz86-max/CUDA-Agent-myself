#include <cuda_runtime.h>
#include <cstdio>

// Kernel to flip a tensor along a single dimension.
// In this task, input is 4D contiguous and dim=2; we hardcode for performance,
// but keep general enough to handle any 4D tensor.
__global__ void flip_single_dim_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t dim,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        // Decode linear index to coordinates (C-order, from strides)
        int64_t remaining = linear;
        int64_t coord0 = remaining / size0;
        remaining %= size0;
        int64_t coord1 = remaining / size1;
        remaining %= size1;
        int64_t coord2 = remaining / size2;
        int64_t coord3 = remaining % size2;

        // Apply flip on the specified dimension
        if (dim == 0) coord0 = size0 - 1 - coord0;
        else if (dim == 1) coord1 = size1 - 1 - coord1;
        else if (dim == 2) coord2 = size2 - 1 - coord2;
        else if (dim == 3) coord3 = size3 - 1 - coord3;

        int64_t src_index = coord0 * stride0 + coord1 * stride1 + coord2 * stride2 + coord3 * stride3;
        output[linear] = input[src_index];
    }
}

// Launcher for flip_forward. Assumes input and output are contiguous tensors
// of same shape, dtype float32, on CUDA. dim is the dimension to flip.
extern "C" void flip_forward_launcher(
    float* output,
    const float* input,
    int64_t dim,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t total_elements,
    cudaStream_t stream)
{
    const int block_size = 256;
    int64_t blocks = (total_elements + block_size - 1) / block_size;
    // Cap grid size to avoid exceeding maximum (2^31-1 in x dimension for compute capability >= 3.0)
    if (blocks > 2147483647) blocks = 2147483647;
    flip_single_dim_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
        output, input, dim,
        size0, size1, size2, size3,
        stride0, stride1, stride2, stride3,
        total_elements);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "flip_forward CUDA error after launch: %s\n", cudaGetErrorString(err));
    }
}
