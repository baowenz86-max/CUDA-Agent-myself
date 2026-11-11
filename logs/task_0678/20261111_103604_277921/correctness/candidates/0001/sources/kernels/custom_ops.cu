// kernels/custom_ops.cu
// Fused flip+cat CUDA implementation: flip along a specified dim then concatenate along same dim.
// Also provides standalone flip and cat kernels for debug/independent use.
// Only CUDA runtime and device code; no torch/ATen/pybind dependencies.

#include <cuda_runtime.h>

// Helper device function to compute linear index from multi-dimensional index given shape.
// ndim <= 4 is assumed (though general code works for any ndim up to 32; using 4 for speed).
__device__ __forceinline__ int64_t idx_to_linear(const int64_t* idx, const int64_t* shape, int ndim) {
    int64_t linear = idx[0];
    #pragma unroll
    for (int i = 1; i < ndim; ++i) {
        linear = linear * shape[i] + idx[i];
    }
    return linear;
}

// Fused flip+cat kernel: output is concatenation of flipped x1 and flipped x2 along dim.
// Each thread writes one output element.
// shape: input shape (both inputs same shape), ndim: number of dims, dim: flip/cat dim (0-indexed, < ndim)
// out: output tensor, x1: first input, x2: second input, total_out: total number of output elements
template <int BLOCK_SIZE>
__global__ void flip_cat_kernel(
    float* __restrict__ out,
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    int64_t total_out,
    const int64_t* __restrict__ shape,
    int ndim,
    int64_t dim
) {
    // Grid-stride loop for large tensors
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total_out; linear += gridDim.x * blockDim.x) {
        // Compute output multi-dimensional index
        int64_t idx[4]; // max dims 4 in this task; could be extended
        int64_t remaining = linear;
        // Compute strides: output shape is same as input except dim doubled, so we need to decompose
        // We'll construct output shape on the fly: shape[dim] * 2
        // For simplicity, precompute output shape in constant memory? We'll just compute indices manually.
        // Since ndim is fixed (4 in this task), we can hardcode for speed, but keep general for flexibility.
        // We'll use a small loop.
        int64_t out_shape[4];
        #pragma unroll
        for (int d = 0; d < ndim; ++d) {
            out_shape[d] = shape[d];
        }
        out_shape[dim] = shape[dim] * 2;

        // Decompose linear into multi-index with output shape
        // Use column-major? We'll use row-major as PyTorch default.
        int64_t temp = remaining;
        for (int d = ndim - 1; d >= 0; --d) {
            idx[d] = temp % out_shape[d];
            temp /= out_shape[d];
        }

        // Determine source: if idx[dim] < shape[dim], source is x1, else x2
        bool from_x1 = idx[dim] < shape[dim];
        const float* src = from_x1 ? x1 : x2;

        // Compute input index: flip along dim: input_dim_idx = shape[dim] - 1 - (idx[dim] % shape[dim])
        int64_t src_idx[4];
        #pragma unroll
        for (int d = 0; d < ndim; ++d) {
            src_idx[d] = idx[d];
        }
        if (from_x1) {
            src_idx[dim] = shape[dim] - 1 - idx[dim];
        } else {
            src_idx[dim] = shape[dim] - 1 - (idx[dim] - shape[dim]);
        }

        // Compute linear source index using input shape
        int64_t src_linear = idx_to_linear(src_idx, shape, ndim);
        out[linear] = src[src_linear];
    }
}

// Standalone flip kernel: flips along dim
// output shape same as input
// shape, ndim, dim as above
template <int BLOCK_SIZE>
__global__ void flip_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int64_t total,
    const int64_t* __restrict__ shape,
    int ndim,
    int64_t dim
) {
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total; linear += gridDim.x * blockDim.x) {
        // Decompose linear into multi-index
        int64_t idx[4];
        int64_t temp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
            idx[d] = temp % shape[d];
            temp /= shape[d];
        }
        // Flip dimension
        idx[dim] = shape[dim] - 1 - idx[dim];
        // Compute linear source index
        int64_t src_linear = idx_to_linear(idx, shape, ndim);
        out[linear] = in[src_linear];
    }
}

// Standalone cat kernel: concatenate two tensors along dim
// Both inputs same shape; output shape is double along dim
// Assumes contiguous inputs and output.
template <int BLOCK_SIZE>
__global__ void cat_kernel(
    float* __restrict__ out,
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    int64_t total_out,
    const int64_t* __restrict__ shape,
    int ndim,
    int64_t dim
) {
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total_out; linear += gridDim.x * blockDim.x) {
        int64_t idx[4];
        int64_t temp = linear;
        // Build output shape with dim doubled
        int64_t out_shape[4];
        #pragma unroll
        for (int d = 0; d < ndim; ++d) {
            out_shape[d] = shape[d];
        }
        out_shape[dim] = shape[dim] * 2;
        for (int d = ndim - 1; d >= 0; --d) {
            idx[d] = temp % out_shape[d];
            temp /= out_shape[d];
        }
        bool from_x1 = idx[dim] < shape[dim];
        const float* src = from_x1 ? x1 : x2;
        // For cat without flip, source index same except dim adjusted
        int64_t src_idx[4];
        #pragma unroll
        for (int d = 0; d < ndim; ++d) {
            src_idx[d] = idx[d];
        }
        if (!from_x1) {
            src_idx[dim] = idx[dim] - shape[dim];
        }
        int64_t src_linear = idx_to_linear(src_idx, shape, ndim);
        out[linear] = src[src_linear];
    }
}

// Launchers: extern "C" functions called from binding.cpp
// flip_cat_forward: dim, x1, x2, plus shape array (int64_t*), ndim, stream
// The binding will allocate output tensor and pass pointers; shape must be provided.
extern "C" void flip_cat_launcher(
    float* out,
    const float* x1,
    const float* x2,
    int64_t total_out,
    const int64_t* shape,
    int ndim,
    int64_t dim,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    int64_t blocks = (total_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Cap grid to reasonable maximum; grid-stride loop handles excess.
    const int64_t max_blocks = 65535; // conservative for all GPUs, can be higher but safe
    if (blocks > max_blocks) blocks = max_blocks;
    flip_cat_kernel<BLOCK_SIZE><<<(unsigned int)blocks, BLOCK_SIZE, 0, stream>>>(
        out, x1, x2, total_out, shape, ndim, dim);
}

extern "C" void flip_launcher(
    float* out,
    const float* in,
    int64_t total,
    const int64_t* shape,
    int ndim,
    int64_t dim,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    int64_t blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64_t max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    flip_kernel<BLOCK_SIZE><<<(unsigned int)blocks, BLOCK_SIZE, 0, stream>>>(
        out, in, total, shape, ndim, dim);
}

extern "C" void cat_launcher(
    float* out,
    const float* x1,
    const float* x2,
    int64_t total_out,
    const int64_t* shape,
    int ndim,
    int64_t dim,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    int64_t blocks = (total_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64_t max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    cat_kernel<BLOCK_SIZE><<<(unsigned int)blocks, BLOCK_SIZE, 0, stream>>>(
        out, x1, x2, total_out, shape, ndim, dim);
}
