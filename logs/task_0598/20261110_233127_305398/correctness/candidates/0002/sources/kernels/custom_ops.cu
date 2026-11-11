#include <cuda_runtime.h>
#include <cstdint>

namespace {

// Compute linear offset from multi-dimensional indices given strides
__device__ __forceinline__ int64_t offset_from_coords(
    const int64_t* strides,
    const int64_t* shape,
    int ndim,
    int64_t linear_index,
    int dim_to_adjust,
    int64_t dim_offset) {
    int64_t offset = 0;
    int64_t remaining = linear_index;
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        if (d >= ndim) break;
        int64_t dim_size = shape[d];
        int64_t coord = remaining % dim_size;
        remaining /= dim_size;
        if (d == dim_to_adjust) {
            coord += dim_offset;
        }
        offset += coord * strides[d];
    }
    return offset;
}

// Kernel: copy input tensor into output tensor along concatenation dimension.
// Each thread handles one element of the input. The input source tensor and
// output tensor strides are passed; dim_offset is added to the coordinate
// along the concatenation dimension for the output.
__global__ void cat_copy_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_elements,
    int ndim,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ output_strides,
    const int64_t* __restrict__ input_shape,
    int64_t dim_to_adjust,
    int64_t dim_offset
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;

    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t in_offset = offset_from_coords(input_strides, input_shape, ndim, i, -1, 0);
        int64_t out_offset = offset_from_coords(output_strides, input_shape, ndim, i, dim_to_adjust, dim_offset);
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

    // Normalize dim if needed (binding already ensures non-negative, but be safe)
    if (dim < 0) dim += ndim;

    // Compute number of elements in x1 and x2
    int64_t x1_elements = 1;
    int64_t x2_elements = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        x1_elements *= x1_shape[d];
        x2_elements *= x2_shape[d];
    }

    int threads = 256;
    int max_blocks = 1073741824; // 2^30, but grid-stride loop allows any large number

    // Launch for x1 with dim_offset=0
    int64_t blocks_x1 = (x1_elements + threads - 1) / threads;
    if (blocks_x1 > max_blocks) blocks_x1 = max_blocks;
    cat_copy_kernel<<<(unsigned int)blocks_x1, threads, 0, stream>>>(
        x1_ptr, out_ptr, x1_elements, ndim,
        x1_strides, out_strides, x1_shape, dim, 0
    );

    // Launch for x2 with dim_offset = x1_shape[dim]
    int64_t blocks_x2 = (x2_elements + threads - 1) / threads;
    if (blocks_x2 > max_blocks) blocks_x2 = max_blocks;
    cat_copy_kernel<<<(unsigned int)blocks_x2, threads, 0, stream>>>(
        x2_ptr, out_ptr, x2_elements, ndim,
        x2_strides, out_strides, x2_shape, dim, x1_shape[dim]
    );
}
