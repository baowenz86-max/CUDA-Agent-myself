#include <cuda_runtime.h>
#include <cstdint>

// Concatenate two float32 tensors along a given dimension (dim >= 0).
// Inputs may have arbitrary strides; output must be contiguous.
// This kernel processes each output linear index via grid-stride loop.
__global__ void concat_kernel(
    float* __restrict__ out,
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    int64_t out_numel,
    int ndim,
    int64_t concat_dim,
    const int64_t* __restrict__ out_shape,
    const int64_t* __restrict__ x1_shape,
    const int64_t* __restrict__ x2_shape,
    const int64_t* __restrict__ x1_strides,
    const int64_t* __restrict__ x2_strides
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t lin = idx; lin < out_numel; lin += stride) {
        // Convert linear index to multidimensional coordinates in output
        int64_t coords[8]; // support up to 8 dimensions (task uses 4)
        int64_t rem = lin;
        for (int d = ndim - 1; d >= 0; --d) {
            coords[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }

        // Determine which input tensor and the coordinate within that tensor
        const float* src_ptr;
        int64_t src_coord[8];
        const int64_t* src_shape;
        const int64_t* src_strides;

        if (coords[concat_dim] < x1_shape[concat_dim]) {
            src_ptr = x1;
            src_shape = x1_shape;
            src_strides = x1_strides;
            for (int d = 0; d < ndim; ++d) {
                src_coord[d] = coords[d];
            }
        } else {
            src_ptr = x2;
            src_shape = x2_shape;
            src_strides = x2_strides;
            for (int d = 0; d < ndim; ++d) {
                src_coord[d] = coords[d];
            }
            src_coord[concat_dim] -= x1_shape[concat_dim];
        }

        // Compute source offset using strides
        int64_t src_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            src_offset += src_coord[d] * src_strides[d];
        }
        out[lin] = src_ptr[src_offset];
    }
}

// Compute max along a given dimension (dim >= 0), removing that dimension.
// Input may have arbitrary strides; output is contiguous.
// Each thread handles one output element (grid-stride loop).
__global__ void amax_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t out_numel,
    int ndim,
    int64_t reduce_dim,
    const int64_t* __restrict__ in_shape,
    const int64_t* __restrict__ in_strides,
    const int64_t* __restrict__ out_shape
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t lin = idx; lin < out_numel; lin += stride) {
        // Convert output linear index to output coordinates (without reduce_dim)
        int64_t out_coords[8];
        int64_t rem = lin;
        for (int d_out = ndim - 2; d_out >= 0; --d_out) {
            out_coords[d_out] = rem % out_shape[d_out];
            rem /= out_shape[d_out];
        }

        // Build input coordinates by inserting a variable index at reduce_dim
        int64_t in_coords[8];
        int d_in = 0;
        for (int d_out = 0; d_out < ndim - 1; ++d_out) {
            if (d_in == reduce_dim) {
                // placeholder, will be filled inside loop
                in_coords[d_in] = 0;
                ++d_in;
            }
            in_coords[d_in] = out_coords[d_out];
            ++d_in;
        }
        if (d_in == reduce_dim) {
            in_coords[d_in] = 0;
            ++d_in;
        }

        float max_val = -CUDART_INF_F;
        int64_t reduce_size = in_shape[reduce_dim];
        for (int64_t k = 0; k < reduce_size; ++k) {
            in_coords[reduce_dim] = k;
            // compute input offset
            int64_t offset = 0;
            for (int d = 0; d < ndim; ++d) {
                offset += in_coords[d] * in_strides[d];
            }
            float val = x[offset];
            if (val > max_val) max_val = val;
        }
        out[lin] = max_val;
    }
}

// Helper: compute output shape for concat
extern "C" void concat_forward_launcher(
    float* out,
    const float* x1,
    const float* x2,
    int ndim,
    int64_t concat_dim,
    const int64_t* out_shape,
    const int64_t* x1_shape,
    const int64_t* x2_shape,
    const int64_t* x1_strides,
    const int64_t* x2_strides,
    int64_t out_numel,
    cudaStream_t stream
) {
    constexpr int block = 256;
    int64_t grid64 = (out_numel + block - 1) / block;
    if (grid64 > 2147483647LL) grid64 = 2147483647LL;
    int grid = static_cast<int>(grid64);
    concat_kernel<<<grid, block, 0, stream>>>(
        out, x1, x2, out_numel, ndim, concat_dim,
        out_shape, x1_shape, x2_shape,
        x1_strides, x2_strides
    );
}

// Helper: compute output shape for amax (dim removed)
extern "C" void amax_forward_launcher(
    const float* x,
    float* out,
    int ndim,
    int64_t reduce_dim,
    const int64_t* in_shape,
    const int64_t* in_strides,
    const int64_t* out_shape,
    int64_t out_numel,
    cudaStream_t stream
) {
    constexpr int block = 256;
    int64_t grid64 = (out_numel + block - 1) / block;
    if (grid64 > 2147483647LL) grid64 = 2147483647LL;
    int grid = static_cast<int>(grid64);
    amax_kernel<<<grid, block, 0, stream>>>(
        x, out, out_numel, ndim, reduce_dim,
        in_shape, in_strides, out_shape
    );
}
