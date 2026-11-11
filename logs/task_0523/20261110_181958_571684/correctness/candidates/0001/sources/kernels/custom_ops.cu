#include <cuda_runtime.h>

// Kernel to flip a tensor along a single dimension.
// Supports arbitrary number of dimensions via sizes and strides.
// Only float32 used in this task.
__global__ void flip_single_dim_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t dim,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ strides,
    int64_t ndim,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        int64_t remaining = linear;
        int64_t src_index = 0;
        for (int64_t d = 0; d < ndim; ++d) {
            int64_t coord = remaining / sizes[d];
            remaining = remaining % sizes[d];
            if (d == dim) {
                coord = sizes[d] - 1 - coord;
            }
            src_index += coord * strides[d];
        }
        output[linear] = input[src_index];
    }
}

// Launcher for flip_forward. Assumes input and output are contiguous tensors
// of same shape, dtype float32, on CUDA. dim is the dimension to flip.
extern "C" void flip_forward_launcher(
    float* output,
    const float* input,
    int64_t dim,
    const int64_t* sizes,
    const int64_t* strides,
    int64_t ndim,
    int64_t total_elements,
    cudaStream_t stream)
{
    const int block_size = 256;
    int64_t blocks = (total_elements + block_size - 1) / block_size;
    // Cap grid size to avoid exceeding maximum (2^31-1 in x dimension for compute capability >= 3.0)
    if (blocks > 2147483647) blocks = 2147483647;
    flip_single_dim_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
        output, input, dim, sizes, strides, ndim, total_elements);
}
