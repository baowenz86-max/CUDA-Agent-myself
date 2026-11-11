#include <cuda_runtime.h>
#include <cstdint>

namespace {

// Compute linear offset from multi-dimensional indices given strides
__device__ __forceinline__ int64_t offset_from_index(
    const int64_t* strides,
    const int64_t* shape,
    int ndim,
    int64_t linear_index) {
    int64_t offset = 0;
    int64_t remaining = linear_index;
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        if (d >= ndim) break;
        int64_t dim_size = shape[d];
        int64_t coord = remaining % dim_size;
        remaining /= dim_size;
        offset += coord * strides[d];
    }
    return offset;
}

// Kernel to copy one input tensor into the output tensor.
// Each thread copies one or more elements (grid-stride loop).
// input: source tensor pointer
// output: destination tensor pointer
// total_elements: number of elements in input tensor
// ndim: number of dimensions
// input_strides, input_shape: for input tensor
// output_strides: for output tensor
// dim: concatenation dimension (already normalized)
// dim_offset: amount to add to the coordinate along dim when computing output offset (0 for first tensor, x1_shape[dim] for second)
__global__ void cat_copy_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_elements,
    int ndim,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ output_strides,
    const int64_t* __restrict__ input_shape,
    int dim,
    int64_t dim_offset
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;

    for (int64_t i = idx; i < total_elements; i += stride) {
        // Compute input offset
        int64_t in_offset = offset_from_index(input_strides, input_shape, ndim, i);

        // Compute output offset using same coordinates, but adjust dim coordinate
        int64_t remaining = i;
        int64_t out_offset = 0;
        #pragma unroll
        for (int d = 0; d < 8; ++d) {
            if (d >= ndim) break;
            int64_t coord = remaining % input_shape[d];
            remaining /= input_shape[d];
            if (d == dim) {
                coord += dim_offset;
            }
            out_offset += coord * output_strides[d];
        }
        output[out_offset] = input[in_offset];
    }
}

}  // namespace

extern "C" void custom_cat_launcher(
    const float* x1_ptr,
    const float* x2_ptr,
    float* out_ptr,
    int64_t ndim,
    const int64_t* x1_shape,
    const int64_t* x2_shape,
    const int64_t* x1_strides,
    const int64_t* x2_strides,
    const int64_t* out_strides,
    int64_t dim,
    int64_t total_output_elems,
    cudaStream_t stream
) {
    if (total_output_elems == 0) return;

    // Normalize dim
    if (dim < 0) dim += ndim;

    // Compute number of elements in each input tensor
    int64_t x1_elements = 1;
    int64_t x2_elements = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        x1_elements *= x1_shape[d];
        x2_elements *= x2_shape[d];
    }

    // Base offsets in output linear indexing for each input's first element
    // For x1, dim_offset = 0 -> base offset 0.
    // For x2, dim_offset = x1_shape[dim], so its first element is at
    // x1_shape[dim] * out_strides[dim].
    // However, our kernel computes out_offset from scratch using coordinates;
    // no separate base offset is added. The dim_offset handles the shift.
    int64_t x1_dim_offset = 0;
    int64_t x2_dim_offset = x1_shape[dim];

    int threads = 256;
    // Grid-stride loop inside kernel, so we can use a reasonable grid size.
    // Use up to 2^31-1 blocks; cap to avoid exceeding max grid dimension.
    const int64_t max_blocks = 1073741824; // 2^30
    int64_t blocks_x1 = (x1_elements + threads - 1) / threads;
    int64_t blocks_x2 = (x2_elements + threads - 1) / threads;
    if (blocks_x1 > max_blocks) blocks_x1 = max_blocks;
    if (blocks_x2 > max_blocks) blocks_x2 = max_blocks;

    // Launch copy for x1
    if (x1_elements > 0) {
        cat_copy_kernel<<<(int)blocks_x1, threads, 0, stream>>>(
            x1_ptr, out_ptr, x1_elements, (int)ndim,
            x1_strides, out_strides, x1_shape, (int)dim, x1_dim_offset
        );
    }

    // Launch copy for x2
    if (x2_elements > 0) {
        cat_copy_kernel<<<(int)blocks_x2, threads, 0, stream>>>(
            x2_ptr, out_ptr, x2_elements, (int)ndim,
            x2_strides, out_strides, x2_shape, (int)dim, x2_dim_offset
        );
    }
}
