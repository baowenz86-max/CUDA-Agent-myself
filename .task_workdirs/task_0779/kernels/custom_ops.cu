// custom_ops.cu
// CUDA kernels and launchers for cat, prod, and fused cat+prod operations.
// All functions operate on float32 tensors with arbitrary strides and produce contiguous outputs.

#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        return; \
    } \
} while (0)

// -----------------------------------------------------------------------------
// Kernel: cat_kernel
// Concatenates two tensors along a given dimension.
// Each thread copies one element from the source (x1 or x2) to the output.
// The source and output are addressed via precomputed index strides for the
// non-concatenated dimensions, while the concatenated dimension is handled by
// offsetting the source index based on whether the element comes from x1 or x2.
// Output tensor is contiguous with shape: all dims same as inputs except dim
// is sum of input dims. Source tensors may be non-contiguous, so we pass their
// strides and use them to compute the linear offset.
// -----------------------------------------------------------------------------
__global__ void cat_kernel(
    const float* __restrict__ src1,
    const float* __restrict__ src2,
    float* __restrict__ dst,
    int dim,
    int ndims,
    const int* __restrict__ shape1,     // shape of x1 (all dims)
    const int* __restrict__ shape2,     // shape of x2 (all dims)
    const int* __restrict__ stride1,    // strides of x1
    const int* __restrict__ stride2,    // strides of x2
    const int* __restrict__ out_shape,  // output shape (contiguous)
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Decode multi-dimensional indices from the flat output index (row-major).
    int indices[8]; // Support up to 8 dimensions
    int64_t remaining = idx;
    for (int d = ndims - 1; d >= 0; --d) {
        indices[d] = remaining % out_shape[d];
        remaining /= out_shape[d];
    }

    // Determine source tensor and adjusted index along dim
    int concat_dim_size1 = shape1[dim];
    int concat_dim_size2 = shape2[dim];
    int concat_idx = indices[dim];

    const float* src;
    const int* shape;
    const int* stride;
    int src_concat_idx;

    if (concat_idx < concat_dim_size1) {
        src = src1;
        shape = shape1;
        stride = stride1;
        src_concat_idx = concat_idx;
    } else {
        src = src2;
        shape = shape2;
        stride = stride2;
        src_concat_idx = concat_idx - concat_dim_size1;
    }

    // Compute source linear index using strides
    int64_t src_linear = 0;
    for (int d = 0; d < ndims; ++d) {
        int idx_d = indices[d];
        if (d == dim) idx_d = src_concat_idx;
        src_linear += (int64_t)idx_d * stride[d];
    }

    dst[idx] = src[src_linear];
}

// -----------------------------------------------------------------------------
// Kernel: prod_kernel
// Computes product along a given dimension.
// Each block reduces one output element (combination of all dims except the
// reduction dim). Grid dimension = number of output elements.
// The kernel loops over the reduction dimension, accumulating product into
// shared memory, then performs block reduction.
// Input may be non-contiguous, so use strides to access.
// -----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void prod_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int dim,
    int ndims,
    const int* __restrict__ shape,   // input shape
    const int* __restrict__ stride,  // input strides
    const int* __restrict__ out_shape, // output shape (dims except dim)
    int64_t out_numel,
    int reduce_size)
{
    int64_t out_idx = blockIdx.x;
    if (out_idx >= out_numel) return;

    // Decode output index into multi-dimensional indices for non-reduced dims
    int indices[8]; // support up to 8 dims
    int64_t remaining = out_idx;
    int out_ndims = ndims - 1;
    // Build indices for non-reduced dims (in original order)
    int non_reduced_indices[8];
    for (int d = 0, od = 0; d < ndims; ++d) {
        if (d == dim) continue;
        // For decoding, we need to account for out_shape order which is the
        // order of non-reduced dims preserved.
        // out_shape[od] corresponds to original dim d.
        // We decode from the end to maintain row-major.
        // Store placeholder; we'll decode after.
        non_reduced_indices[od] = 0; // placeholder
        od++;
    }
    // Decode out_idx into non_reduced_indices in row-major of output tensor
    // output shape is out_shape with ndims-1 entries.
    // We need to fill non_reduced_indices for each original dimension except dim.
    int64_t rem = out_idx;
    for (int od = out_ndims - 1; od >= 0; --od) {
        int original_dim = -1;
        // Find which original dimension corresponds to od-th non-reduced dim
        int count = 0;
        for (int d = 0; d < ndims; ++d) {
            if (d == dim) continue;
            if (count == od) {
                original_dim = d;
                break;
            }
            count++;
        }
        // out_shape[od] is size of that dim
        non_reduced_indices[od] = rem % out_shape[od];
        rem /= out_shape[od];
    }

    // Now compute reduction
    __shared__ float sdata[BLOCK_SIZE];
    float prod = 1.0f;

    // Grid-stride loop over reduction dimension
    for (int i = threadIdx.x; i < reduce_size; i += BLOCK_SIZE) {
        // Build full input indices: start with non-reduced indices
        int full_indices[8];
        int od = 0;
        for (int d = 0; d < ndims; ++d) {
            if (d == dim) {
                full_indices[d] = i;
            } else {
                full_indices[d] = non_reduced_indices[od++];
            }
        }
        int64_t src_linear = 0;
        for (int d = 0; d < ndims; ++d) {
            src_linear += (int64_t)full_indices[d] * stride[d];
        }
        prod *= input[src_linear];
    }

    // Block reduction
    sdata[threadIdx.x] = prod;
    __syncthreads();

    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] *= sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[out_idx] = sdata[0];
    }
}

// -----------------------------------------------------------------------------
// Kernel: cat_prod_fused_kernel
// Directly computes product over concatenated dimension without materializing
// the concatenated tensor. Each block handles one output element.
// The reduction size is combined = shape1[dim] + shape2[dim].
// For indices < shape1[dim], data from x1; else from x2.
// -----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void cat_prod_fused_kernel(
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    float* __restrict__ output,
    int dim,
    int ndims,
    const int* __restrict__ shape1,
    const int* __restrict__ shape2,
    const int* __restrict__ stride1,
    const int* __restrict__ stride2,
    const int* __restrict__ out_shape, // output shape (dims except dim)
    int64_t out_numel,
    int reduce_size1,
    int reduce_size2)
{
    int64_t out_idx = blockIdx.x;
    if (out_idx >= out_numel) return;

    // Decode output index into non-reduced dims similar to prod_kernel
    int out_ndims = ndims - 1;
    int non_reduced_indices[8]; // for original dims except dim, indexed by od
    int64_t rem = out_idx;
    for (int od = out_ndims - 1; od >= 0; --od) {
        // find original dim for od
        int original_dim = -1;
        int count = 0;
        for (int d = 0; d < ndims; ++d) {
            if (d == dim) continue;
            if (count == od) {
                original_dim = d;
                break;
            }
            count++;
        }
        non_reduced_indices[od] = rem % out_shape[od];
        rem /= out_shape[od];
    }

    __shared__ float sdata[BLOCK_SIZE];
    float prod = 1.0f;

    // First part: x1
    for (int i = threadIdx.x; i < reduce_size1; i += BLOCK_SIZE) {
        int full_indices[8];
        int od = 0;
        for (int d = 0; d < ndims; ++d) {
            if (d == dim) full_indices[d] = i;
            else full_indices[d] = non_reduced_indices[od++];
        }
        int64_t src_linear = 0;
        for (int d = 0; d < ndims; ++d) {
            src_linear += (int64_t)full_indices[d] * stride1[d];
        }
        prod *= x1[src_linear];
    }

    // Second part: x2
    for (int i = threadIdx.x; i < reduce_size2; i += BLOCK_SIZE) {
        int full_indices[8];
        int od = 0;
        for (int d = 0; d < ndims; ++d) {
            if (d == dim) full_indices[d] = i;
            else full_indices[d] = non_reduced_indices[od++];
        }
        int64_t src_linear = 0;
        for (int d = 0; d < ndims; ++d) {
            src_linear += (int64_t)full_indices[d] * stride2[d];
        }
        prod *= x2[src_linear];
    }

    sdata[threadIdx.x] = prod;
    __syncthreads();

    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] *= sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        output[out_idx] = sdata[0];
    }
}

// -----------------------------------------------------------------------------
// Host launchers (extern "C")
// These functions assume all input tensors are on CUDA, float32, and valid.
// They do NOT perform any tensor checks; the binding layer is responsible for
// ensuring correctness and allocating output tensors and copying shape/stride
// arrays to device memory.
//------------------------------------------------------------------------------

// Helper to launch cat kernel with dynamic block size and grid size
void launch_cat(
    cudaStream_t stream,
    const float* src1,
    const float* src2,
    float* dst,
    int dim,
    int ndims,
    const int* shape1,
    const int* shape2,
    const int* stride1,
    const int* stride2,
    const int* out_shape,
    int64_t total_elements)
{
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    cat_kernel<<<grid_size, block_size, 0, stream>>>(
        src1, src2, dst, dim, ndims,
        shape1, shape2, stride1, stride2, out_shape,
        total_elements);
}

// Helper to launch prod kernel
void launch_prod(
    cudaStream_t stream,
    const float* input,
    float* output,
    int dim,
    int ndims,
    const int* shape,
    const int* stride,
    const int* out_shape,
    int64_t out_numel,
    int reduce_size)
{
    const int block_size = 256;
    int grid_size = out_numel;
    prod_kernel<block_size><<<grid_size, block_size, 0, stream>>>(
        input, output, dim, ndims, shape, stride, out_shape,
        out_numel, reduce_size);
}

// Helper to launch fused cat+prod kernel
void launch_cat_prod_fused(
    cudaStream_t stream,
    const float* x1,
    const float* x2,
    float* output,
    int dim,
    int ndims,
    const int* shape1,
    const int* shape2,
    const int* stride1,
    const int* stride2,
    const int* out_shape,
    int64_t out_numel,
    int reduce_size1,
    int reduce_size2)
{
    const int block_size = 256;
    int grid_size = out_numel;
    cat_prod_fused_kernel<block_size><<<grid_size, block_size, 0, stream>>>(
        x1, x2, output, dim, ndims,
        shape1, shape2, stride1, stride2, out_shape,
        out_numel, reduce_size1, reduce_size2);
}

// -----------------------------------------------------------------------------
// Exported launcher functions for binding layer
// The binding layer will handle tensor allocation, dimension checks, and
// copying shape/stride arrays to device (or use a different approach).
// Here we provide functions that accept raw pointers and precomputed metadata.
// All shape/stride arrays are assumed to reside in device memory, passed as
// constant pointers to device memory.
// -----------------------------------------------------------------------------

extern "C" void cat_forward_launcher(
    const float* src1,
    const float* src2,
    float* dst,
    int dim,
    int ndims,
    const int* shape1,
    const int* shape2,
    const int* stride1,
    const int* stride2,
    const int* out_shape,
    int64_t total_elements,
    cudaStream_t stream)
{
    launch_cat(stream, src1, src2, dst, dim, ndims,
               shape1, shape2, stride1, stride2, out_shape, total_elements);
}

extern "C" void prod_forward_launcher(
    const float* input,
    float* output,
    int dim,
    int ndims,
    const int* shape,
    const int* stride,
    const int* out_shape,
    int64_t out_numel,
    int reduce_size,
    cudaStream_t stream)
{
    launch_prod(stream, input, output, dim, ndims,
                shape, stride, out_shape, out_numel, reduce_size);
}

extern "C" void cat_prod_forward_launcher(
    const float* x1,
    const float* x2,
    float* output,
    int dim,
    int ndims,
    const int* shape1,
    const int* shape2,
    const int* stride1,
    const int* stride2,
    const int* out_shape,
    int64_t out_numel,
    int reduce_size1,
    int reduce_size2,
    cudaStream_t stream)
{
    launch_cat_prod_fused(stream, x1, x2, output, dim, ndims,
                          shape1, shape2, stride1, stride2, out_shape,
                          out_numel, reduce_size1, reduce_size2);
}
