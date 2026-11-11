#include <cuda_runtime.h>

// CUDA kernel for custom transpose: swap dimension dim with last dimension.
// Input and output data pointers are both raw float pointers; shapes and strides are passed explicitly.
// Output tensor is non-contiguous view in PyTorch, but here we materialize the transposed tensor in a new allocation with
// contiguous memory (the binding layer will allocate output with appropriate size; we just write elements in transposed order).
// However, contract says output should be a view (non-contiguous). But since we return a newly allocated tensor via binding,
// we must ensure it has the correct shape and strides. The launcher only writes data; the binding layer will create the tensor
// with desired strides. We assume output pointer points to a buffer of size total elements, and we write data in contiguous order.
__global__ void transpose_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  const long long* __restrict__ input_shape,
                                  const long long* __restrict__ input_strides,
                                  int ndim,
                                  int dim,
                                  size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t linear_idx = idx; linear_idx < total_elements; linear_idx += stride) {
        // Convert linear index to multi-dimensional coordinates in output (transposed layout)
        size_t remaining = linear_idx;
        long long out_coords[8]; // assume max 8 dims, ndim <= 8 for this task
        // output shape is same as input shape but with dim and last dim swapped
        long long out_shape[8];
        for (int i = 0; i < ndim; ++i) {
            out_shape[i] = input_shape[i];
        }
        // swap dim and ndim-1
        long long tmp = out_shape[dim];
        out_shape[dim] = out_shape[ndim - 1];
        out_shape[ndim - 1] = tmp;

        // compute output coordinates (row-major)
        for (int d = ndim - 1; d >= 0; --d) {
            out_coords[d] = remaining % out_shape[d];
            remaining /= out_shape[d];
        }
        // Now map to input coordinates by swapping dim and last dim
        long long in_coords[8];
        for (int d = 0; d < ndim; ++d) {
            in_coords[d] = out_coords[d];
        }
        long long tmp_coord = in_coords[dim];
        in_coords[dim] = in_coords[ndim - 1];
        in_coords[ndim - 1] = tmp_coord;

        // compute input linear offset using input strides
        size_t input_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            input_offset += in_coords[d] * input_strides[d];
        }
        output[linear_idx] = input[input_offset];
    }
}

// CUDA kernel for custom cat 2x along last dimension.
// Input tensor has shape [..., C] and we need to output [..., 2*C] with first C columns being original and next C being copy.
// Input may be non-contiguous (after transpose in this task, but we handle general strides).
// Output is assumed contiguous.
__global__ void cat2x_kernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              const long long* __restrict__ input_shape,
                              const long long* __restrict__ input_strides,
                              int ndim,
                              size_t total_output_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    long long last_dim_size = input_shape[ndim - 1];
    for (size_t linear_idx = idx; linear_idx < total_output_elements; linear_idx += stride) {
        // Determine output coordinates (contiguous)
        size_t remaining = linear_idx;
        long long out_coords[8];
        // output shape is same as input_shape but last dim doubled
        long long out_shape[8];
        for (int d = 0; d < ndim; ++d) {
            out_shape[d] = input_shape[d];
        }
        out_shape[ndim - 1] *= 2;
        for (int d = ndim - 1; d >= 0; --d) {
            out_coords[d] = remaining % out_shape[d];
            remaining /= out_shape[d];
        }
        // Map to input coordinates: last coordinate mod last_dim_size
        long long in_coords[8];
        for (int d = 0; d < ndim; ++d) {
            in_coords[d] = out_coords[d];
        }
        in_coords[ndim - 1] = out_coords[ndim - 1] % last_dim_size;

        size_t input_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            input_offset += in_coords[d] * input_strides[d];
        }
        output[linear_idx] = input[input_offset];
    }
}

// Host launcher for transpose.
// Parameters:
//   input: device pointer to input tensor data (float32)
//   output: device pointer to output tensor data (float32) - must have space for total elements
//   input_shape: host pointer to array of shape (ndim int64 values)
//   input_strides: host pointer to array of strides (ndim int64 values)
//   ndim: number of dimensions (scalar)
//   dim: the dimension to swap with last (scalar)
//   total_elements: total number of elements in output (same as input)
//   stream: CUDA stream
// We copy shape and strides to device constant memory or pass as kernel arguments? We'll copy to device memory.
extern "C" void custom_transpose_launcher(const float* input, float* output,
                                           const int64_t* input_shape, const int64_t* input_strides,
                                           int ndim, int dim, size_t total_elements,
                                           cudaStream_t stream) {
    // Allocate device memory for shape and strides (small)
    int64_t* d_shape;
    int64_t* d_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpyAsync(d_shape, input_shape, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_strides, input_strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid cap; kernel has grid-stride loop
    transpose_kernel<<<blocks, threads, 0, stream>>>(input, output, d_shape, d_strides, ndim, dim, total_elements);

    cudaFree(d_shape);
    cudaFree(d_strides);
}

// Host launcher for cat 2x along last dimension.
extern "C" void custom_cat_2x_launcher(const float* input, float* output,
                                         const int64_t* input_shape, const int64_t* input_strides,
                                         int ndim, size_t total_output_elements,
                                         cudaStream_t stream) {
    int64_t* d_shape;
    int64_t* d_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpyAsync(d_shape, input_shape, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_strides, input_strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (total_output_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    cat2x_kernel<<<blocks, threads, 0, stream>>>(input, output, d_shape, d_strides, ndim, total_output_elements);

    cudaFree(d_shape);
    cudaFree(d_strides);
}
