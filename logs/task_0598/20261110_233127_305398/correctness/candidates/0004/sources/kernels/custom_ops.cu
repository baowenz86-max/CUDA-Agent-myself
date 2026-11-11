#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ int64_t offset_from_index(
    const int64_t* strides,
    const int64_t* shape,
    int ndim,
    int64_t linear_index,
    int concat_dim,
    int64_t dim_offset,
    const int64_t* out_strides
) {
    int64_t offset = 0;
    int64_t remaining = linear_index;
    int64_t coord[8];
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        if (d >= ndim) break;
        int64_t dim_size = shape[d];
        coord[d] = remaining % dim_size;
        remaining /= dim_size;
    }
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        if (d >= ndim) break;
        int64_t c = coord[d];
        if (d == concat_dim) c += dim_offset;
        offset += c * out_strides[d];
    }
    return offset;
}

__global__ void cat_copy_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_elements,
    int ndim,
    int concat_dim,
    int64_t dim_offset,
    const int64_t* __restrict__ output_strides,
    const int64_t* __restrict__ input_shape,
    const int64_t* __restrict__ input_strides
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t in_offset = offset_from_index(input_strides, input_shape, ndim, i, concat_dim, 0, input_strides);
        int64_t out_offset = offset_from_index(input_strides, input_shape, ndim, i, concat_dim, dim_offset, output_strides);
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
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) return;

    int64_t x1_elements = 1;
    int64_t x2_elements = 1;
    for (int d = 0; d < ndim; ++d) {
        x1_elements *= x1_shape[d];
        x2_elements *= x2_shape[d];
    }

    int threads = 256;
    int64_t blocks_x1 = (x1_elements + threads - 1) / threads;
    int64_t blocks_x2 = (x2_elements + threads - 1) / threads;
    int max_blocks = 1073741824;
    if (blocks_x1 > max_blocks) blocks_x1 = max_blocks;
    if (blocks_x2 > max_blocks) blocks_x2 = max_blocks;

    cat_copy_kernel<<<(int)blocks_x1, threads, 0, stream>>>(
        x1_ptr, out_ptr, x1_elements, (int)ndim, (int)dim, 0,
        out_strides, x1_shape, x1_strides);

    cat_copy_kernel<<<(int)blocks_x2, threads, 0, stream>>>(
        x2_ptr, out_ptr, x2_elements, (int)ndim, (int)dim, x1_shape[dim],
        out_strides, x2_shape, x2_strides);
}
