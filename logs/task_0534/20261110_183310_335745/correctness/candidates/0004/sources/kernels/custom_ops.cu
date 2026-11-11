// CUDA kernels for max_dim_keepdim and cosine_similarity_dim
#include <cuda_runtime.h>
#include <float.h>

// Kernel for max reduction along a specified dimension.
// Each block handles one output element.
__global__ void max_reduce_kernel(const float* __restrict__ input,
                                  float* __restrict__ output_values,
                                  int64_t* __restrict__ output_indices,
                                  const int64_t* __restrict__ input_shape,
                                  const int64_t* __restrict__ input_strides,
                                  const int64_t* __restrict__ output_shape,
                                  const int64_t* __restrict__ output_strides,
                                  int ndim,
                                  int dim,
                                  int dim_size,
                                  int64_t outer_size,
                                  int64_t inner_size,
                                  int output_ndim)
{
    int64_t out_idx = blockIdx.x;
    if (out_idx >= outer_size * inner_size) return;

    int64_t outer = out_idx / inner_size;
    int64_t inner = out_idx % inner_size;

    // Compute input base offset for this output element.
    int64_t input_base = 0;
    int64_t rem = outer;
    for (int i = 0; i < dim; i++) {
        int64_t extent = input_shape[i];
        int64_t coord = rem % extent;
        rem /= extent;
        input_base += coord * input_strides[i];
    }
    rem = inner;
    for (int i = ndim - 1; i > dim; i--) {
        int64_t extent = input_shape[i];
        int64_t coord = rem % extent;
        rem /= extent;
        input_base += coord * input_strides[i];
    }
    int64_t stride_dim = input_strides[dim];

    float max_val = -FLT_MAX;
    int64_t max_idx = 0;
    for (int64_t i = 0; i < dim_size; ++i) {
        float val = input[input_base + i * stride_dim];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }

    // Compute output offset using output_shape and output_strides.
    // output_ndim may be ndim or ndim-1 depending on keepdim.
    int64_t output_offset = 0;
    rem = outer;
    for (int i = 0; i < dim; i++) {
        int64_t extent = output_shape[i];
        int64_t coord = rem % extent;
        rem /= extent;
        output_offset += coord * output_strides[i];
    }
    rem = inner;
    for (int i = ndim - 1; i > dim; i--) {
        int out_i = (output_ndim < ndim) ? (i - 1) : i;
        int64_t extent = output_shape[out_i];
        int64_t coord = rem % extent;
        rem /= extent;
        output_offset += coord * output_strides[out_i];
    }
    output_values[output_offset] = max_val;
    output_indices[output_offset] = max_idx;
}

// Launcher for max reduction.
extern "C" void max_dim_keepdim_launcher(
    const float* input,
    float* output_values,
    int64_t* output_indices,
    const int64_t* input_shape,
    const int64_t* input_strides,
    const int64_t* output_shape,
    const int64_t* output_strides,
    int ndim,
    int64_t dim,
    bool keepdim,
    cudaStream_t stream)
{
    if (dim < 0) dim += ndim;
    int64_t dim_size = input_shape[dim];
    int64_t outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= input_shape[i];
    int64_t inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) inner_size *= input_shape[i];
    int64_t total_out = outer_size * inner_size;

    int output_ndim = keepdim ? ndim : ndim - 1;

    dim3 grid(total_out);
    dim3 block(1);
    max_reduce_kernel<<<grid, block, 0, stream>>>(
        input, output_values, output_indices,
        input_shape, input_strides, output_shape, output_strides,
        ndim, dim, dim_size, outer_size, inner_size, output_ndim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return;
}

// --------------------------- Cosine Similarity Kernel ---------------------------
// Compute cosine similarity along dimension dim with broadcasting.
__global__ void cosine_similarity_kernel(
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    float* __restrict__ output,
    const int64_t* __restrict__ x1_shape,
    const int64_t* __restrict__ x1_strides,
    const int64_t* __restrict__ x2_shape,
    const int64_t* __restrict__ x2_strides,
    const int64_t* __restrict__ output_shape,
    const int64_t* __restrict__ output_strides,
    int ndim,
    int dim,
    int dim_size,
    int64_t outer_size,
    int64_t inner_size)
{
    int64_t out_idx = blockIdx.x;
    if (out_idx >= outer_size * inner_size) return;

    int64_t outer = out_idx / inner_size;
    int64_t inner = out_idx % inner_size;

    // Compute base offsets for x1 and x2 for this output element.
    int64_t x1_base = 0;
    int64_t rem = outer;
    for (int i = 0; i < dim; i++) {
        int64_t extent = x1_shape[i];
        int64_t coord = rem % extent;
        rem /= extent;
        x1_base += coord * x1_strides[i];
    }
    rem = inner;
    for (int i = ndim - 1; i > dim; i--) {
        int64_t extent = x1_shape[i];
        int64_t coord = rem % extent;
        rem /= extent;
        x1_base += coord * x1_strides[i];
    }
    int64_t x1_stride_dim = x1_strides[dim];

    // x2 base: handle broadcasting, but we assume same ndim.
    int64_t x2_base = 0;
    rem = outer;
    for (int i = 0; i < dim; i++) {
        int64_t extent = x2_shape[i];
        int64_t coord = rem % extent;
        rem /= extent;
        x2_base += coord * x2_strides[i];
    }
    rem = inner;
    for (int i = ndim - 1; i > dim; i--) {
        int64_t extent = x2_shape[i];
        int64_t coord = rem % extent;
        rem /= extent;
        x2_base += coord * x2_strides[i];
    }
    int64_t x2_stride_dim = x2_strides[dim];
    if (x2_shape[dim] == 1) {
        x2_stride_dim = 0;
    }

    float dot = 0.f;
    float norm1 = 0.f;
    float norm2 = 0.f;
    for (int64_t i = 0; i < dim_size; ++i) {
        float v1 = x1[x1_base + i * x1_stride_dim];
        float v2 = x2[x2_base + i * x2_stride_dim];
        dot += v1 * v2;
        norm1 += v1 * v1;
        norm2 += v2 * v2;
    }
    const float eps = 1e-8f;
    float denom = sqrtf(norm1) * sqrtf(norm2) + eps;
    float result = dot / denom;

    // Compute output offset using output_shape and output_strides.
    // output_shape has ndim-1 dimensions (dim removed).
    int64_t output_offset = 0;
    rem = outer;
    for (int i = 0; i < dim; i++) {
        int64_t extent = output_shape[i];
        int64_t coord = rem % extent;
        rem /= extent;
        output_offset += coord * output_strides[i];
    }
    rem = inner;
    for (int i = ndim - 1; i > dim; i--) {
        int out_i = i - 1;
        int64_t extent = output_shape[out_i];
        int64_t coord = rem % extent;
        rem /= extent;
        output_offset += coord * output_strides[out_i];
    }
    output[output_offset] = result;
}

// Launcher for cosine similarity.
extern "C" void cosine_similarity_dim_launcher(
    const float* x1,
    const float* x2,
    float* output,
    const int64_t* x1_shape,
    const int64_t* x1_strides,
    const int64_t* x2_shape,
    const int64_t* x2_strides,
    const int64_t* output_shape,
    const int64_t* output_strides,
    int ndim,
    int64_t dim,
    cudaStream_t stream)
{
    if (dim < 0) dim += ndim;
    int64_t dim_size = x1_shape[dim];
    int64_t outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= x1_shape[i];
    int64_t inner_size = 1;
    for (int i = dim + 1; i < ndim; ++i) inner_size *= x1_shape[i];
    int64_t total_out = outer_size * inner_size;

    dim3 grid(total_out);
    dim3 block(1);
    cosine_similarity_kernel<<<grid, block, 0, stream>>>(
        x1, x2, output,
        x1_shape, x1_strides, x2_shape, x2_strides, output_shape, output_strides,
        ndim, dim, dim_size, outer_size, inner_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return;
}
