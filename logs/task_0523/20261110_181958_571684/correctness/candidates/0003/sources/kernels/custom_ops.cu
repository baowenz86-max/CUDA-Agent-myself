#include <cuda_runtime.h>

// Kernel to flip a tensor along a single dimension.
// Fixed for up to 4 dimensions (as in this task).
// Input tensor is contiguous; sizes/strides passed by value.
__global__ void flip_single_dim_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t dim,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t ndim,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        int64_t remaining = linear;
        int64_t src_index = 0;

        // Manually unroll for up to 4 dims
        if (ndim >= 4) {
            int64_t coord3 = remaining / size3;
            remaining = remaining % size3;
            if (dim == 3) coord3 = size3 - 1 - coord3;
            src_index += coord3 * stride3;
        }
        if (ndim >= 3) {
            int64_t coord2 = remaining / size2;
            remaining = remaining % size2;
            if (dim == 2) coord2 = size2 - 1 - coord2;
            src_index += coord2 * stride2;
        }
        if (ndim >= 2) {
            int64_t coord1 = remaining / size1;
            remaining = remaining % size1;
            if (dim == 1) coord1 = size1 - 1 - coord1;
            src_index += coord1 * stride1;
        }
        if (ndim >= 1) {
            int64_t coord0 = remaining;
            if (dim == 0) coord0 = size0 - 1 - coord0;
            src_index += coord0 * stride0;
        }

        output[linear] = input[src_index];
    }
}

// Launcher for flip_forward.
// Assumes input and output are contiguous tensors of same shape, dtype float32, on CUDA.
// dim is the dimension to flip (already normalized to non-negative).
extern "C" void flip_forward_launcher(
    float* output,
    const float* input,
    int64_t dim,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t ndim,
    int64_t total_elements,
    cudaStream_t stream)
{
    const int block_size = 256;
    int64_t blocks = (total_elements + block_size - 1) / block_size;
    // Cap grid size to avoid exceeding maximum (2^31-1 in x dimension)
    if (blocks > 2147483647) blocks = 2147483647;
    flip_single_dim_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
        output, input, dim,
        size0, size1, size2, size3,
        stride0, stride1, stride2, stride3,
        ndim, total_elements);
}
