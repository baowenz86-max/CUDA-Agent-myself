#include <cuda_runtime.h>

// Kernel to flip a tensor along a single dimension.
// sizes and strides are passed as scalar values because ndim=4 in this task.
// The kernel is specialized for ndim=4 but can be extended if needed.
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
        int64_t remaining = linear;
        int64_t coord0 = remaining / size0;
        remaining = remaining % size0;
        int64_t coord1 = remaining / size1;
        remaining = remaining % size1;
        int64_t coord2 = remaining / size2;
        remaining = remaining % size2;
        int64_t coord3 = remaining; // size3 = width, remaining < size3

        if (dim == 0) coord0 = size0 - 1 - coord0;
        if (dim == 1) coord1 = size1 - 1 - coord1;
        if (dim == 2) coord2 = size2 - 1 - coord2;
        if (dim == 3) coord3 = size3 - 1 - coord3;

        int64_t src_index = coord0 * stride0 + coord1 * stride1 + coord2 * stride2 + coord3 * stride3;
        output[linear] = input[src_index];
    }
}

// Launcher for flip_forward. Assumes input and output are contiguous tensors
// of same shape, dtype float32, on CUDA. dim is the dimension to flip.
// Here we expect ndim == 4; if not, this launcher could be extended.
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
    // Cap grid size to avoid exceeding maximum grid dimension (2^31-1).
    if (blocks > 2147483647) blocks = 2147483647;
    flip_single_dim_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
        output, input, dim,
        size0, size1, size2, size3,
        stride0, stride1, stride2, stride3,
        total_elements);
}
