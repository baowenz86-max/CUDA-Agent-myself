// kernels/custom_ops.cu
// Fused flip+cat CUDA implementation: flip along a specified dim then concatenate along same dim.
// Also provides standalone flip and cat kernels for debug/independent use.
// Only CUDA runtime and device code; no torch/ATen/pybind dependencies.

#include <cuda_runtime.h>

// Helper device function to compute linear index from multi-dimensional index given shape.
// ndim is assumed to be 3 or 4 in this task; we use a fixed max dims 4 for speed.
__device__ __forceinline__ int64_t idx_to_linear4(const int64_t idx[4], const int64_t shape[4], int ndim) {
    int64_t linear = idx[0];
    #pragma unroll
    for (int i = 1; i < ndim; ++i) {
        linear = linear * shape[i] + idx[i];
    }
    return linear;
}

// Fused flip+cat kernel: output is concatenation of flipped x1 and flipped x2 along dim.
// Each thread writes one output element.
// shape array (fixed to 4 int64_t values) contains input shape; ndim is number of dimensions used (<=4);
// dim is flip/cat dim (0-indexed).
// out: output tensor, x1: first input, x2: second input, total_out: total number of output elements.
template <int BLOCK_SIZE>
__global__ void flip_cat_kernel(
    float* __restrict__ out,
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    int64_t total_out,
    const int64_t* __restrict__ shape,  // device pointer (not used now; we'll pass by value? Actually we pass scalar args below)
    int ndim,
    int64_t dim
) {
    // Grid-stride loop
    for (int64_t linear = blockIdx.x * blockDim.x + threadIdx.x; linear < total_out; linear += gridDim.x * blockDim.x) {
        // Compute output multi-dimensional index using output shape (with dim doubled)
        int64_t idx[4];
        int64_t out_shape[4];
        #pragma unroll
        for (int d = 0; d < ndim; ++d) {
            out_shape[d] = shape[d];
        }
        out_shape[dim] = shape[dim] * 2;

        int64_t temp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
            idx[d] = temp % out_shape[d];
            temp /= out_shape[d];
        }

        // Determine source: if idx[dim] < shape[dim], source is x1, else x2
        bool from_x1 = idx[dim] < shape[dim];
        const float* src = from_x1 ? x1 : x2;

        // Compute input index: flip along dim
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

        int64_t src_linear = idx_to_linear4(src_idx, shape, ndim);
        out[linear] = src[src_linear];
    }
}

// Standalone flip kernel
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
        int64_t idx[4];
        int64_t temp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
            idx[d] = temp % shape[d];
            temp /= shape[d];
        }
        idx[dim] = shape[dim] - 1 - idx[dim];
        int64_t src_linear = idx_to_linear4(idx, shape, ndim);
        out[linear] = in[src_linear];
    }
}

// Standalone cat kernel
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
        int64_t out_shape[4];
        #pragma unroll
        for (int d = 0; d < ndim; ++d) {
            out_shape[d] = shape[d];
        }
        out_shape[dim] = shape[dim] * 2;
        int64_t temp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
            idx[d] = temp % out_shape[d];
            temp /= out_shape[d];
        }
        bool from_x1 = idx[dim] < shape[dim];
        const float* src = from_x1 ? x1 : x2;
        int64_t src_idx[4];
        #pragma unroll
        for (int d = 0; d < ndim; ++d) {
            src_idx[d] = idx[d];
        }
        if (!from_x1) {
            src_idx[dim] = idx[dim] - shape[dim];
        }
        int64_t src_linear = idx_to_linear4(src_idx, shape, ndim);
        out[linear] = src[src_linear];
    }
}

// Launchers with shape passed as scalar array (by value) to avoid host pointer issue.
// We'll use a fixed-size array of 4 int64_t values. The binding will copy shape into such an array.
extern "C" void flip_cat_launcher(
    float* out,
    const float* x1,
    const float* x2,
    int64_t total_out,
    const int64_t shape[4],  // by value array? In C, array parameter decays to pointer, so we need to pass explicitly as values.
    int ndim,
    int64_t dim,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    int64_t blocks = (total_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64_t max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    // Need to copy shape to device constant memory or pass as kernel arguments? Since array parameter decays to pointer, we cannot pass by value easily.
    // Instead, we will create a device copy or use a kernel that takes shape as four scalar arguments.
    // Let's change kernels to take shape as four separate int64_t values (or three for 3D?) but we have 4D fixed.
    // Since we know ndim <=4, we can pass shape0, shape1, shape2, shape3 (with unused zeros).
    // However for simplicity and correctness, we'll allocate device memory inside launcher and copy shape, then pass pointer.
    // This ensures shape pointer is valid on device.
    int64_t* d_shape;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMemcpyAsync(d_shape, shape, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    flip_cat_kernel<BLOCK_SIZE><<<(unsigned int)blocks, BLOCK_SIZE, 0, stream>>>(
        out, x1, x2, total_out, d_shape, ndim, dim);
    cudaFree(d_shape); // This is asynchronous free? Safer to not free or use cudaFree after kernel. But free after launch may be okay if stream ordered? Actually cudaFree synchronizes? Better use cudaFree after launch, but it may sync. We'll use cudaFree after kernel, but it might cause sync; acceptable for correctness.
    // Better: we can allocate d_shape once? But for simplicity, use cudaMalloc/cudaMemcpy/cudaFree each launch.
    // Note: cudaFree after kernel launch might block until kernel completes, which is fine.
    // We could also avoid allocation by passing shape as kernel parameters (each as int64_t), but array is easier.
    // To avoid repeated allocations, we could use constant memory or __constant__ array, but binding cannot set it easily.
    // We'll keep allocation and free.
}

extern "C" void flip_launcher(
    float* out,
    const float* in,
    int64_t total,
    const int64_t shape[4],
    int ndim,
    int64_t dim,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    int64_t blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64_t max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    int64_t* d_shape;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMemcpyAsync(d_shape, shape, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    flip_kernel<BLOCK_SIZE><<<(unsigned int)blocks, BLOCK_SIZE, 0, stream>>>(
        out, in, total, d_shape, ndim, dim);
    cudaFree(d_shape);
}

extern "C" void cat_launcher(
    float* out,
    const float* x1,
    const float* x2,
    int64_t total_out,
    const int64_t shape[4],
    int ndim,
    int64_t dim,
    cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    int64_t blocks = (total_out + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int64_t max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    int64_t* d_shape;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMemcpyAsync(d_shape, shape, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cat_kernel<BLOCK_SIZE><<<(unsigned int)blocks, BLOCK_SIZE, 0, stream>>>(
        out, x1, x2, total_out, d_shape, ndim, dim);
    cudaFree(d_shape);
}
