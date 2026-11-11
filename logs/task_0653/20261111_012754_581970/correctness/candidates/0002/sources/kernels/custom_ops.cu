#include <cuda_runtime.h>
#include <cstdint>

__global__ void transpose_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  const int64_t* __restrict__ input_shape,
                                  const int64_t* __restrict__ input_strides,
                                  int ndim,
                                  int dim,
                                  size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t linear_idx = idx; linear_idx < total_elements; linear_idx += stride) {
        size_t remaining = linear_idx;
        int64_t out_coords[8];
        int64_t out_shape[8];
        for (int i = 0; i < ndim; ++i) {
            out_shape[i] = input_shape[i];
        }
        int64_t tmp = out_shape[dim];
        out_shape[dim] = out_shape[ndim - 1];
        out_shape[ndim - 1] = tmp;

        for (int d = ndim - 1; d >= 0; --d) {
            out_coords[d] = remaining % out_shape[d];
            remaining /= out_shape[d];
        }
        int64_t in_coords[8];
        for (int d = 0; d < ndim; ++d) {
            in_coords[d] = out_coords[d];
        }
        int64_t tmp_coord = in_coords[dim];
        in_coords[dim] = in_coords[ndim - 1];
        in_coords[ndim - 1] = tmp_coord;

        size_t input_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            input_offset += in_coords[d] * input_strides[d];
        }
        output[linear_idx] = input[input_offset];
    }
}

__global__ void cat2x_kernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              const int64_t* __restrict__ input_shape,
                              const int64_t* __restrict__ input_strides,
                              int ndim,
                              size_t total_output_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    int64_t last_dim_size = input_shape[ndim - 1];
    for (size_t linear_idx = idx; linear_idx < total_output_elements; linear_idx += stride) {
        size_t remaining = linear_idx;
        int64_t out_coords[8];
        int64_t out_shape[8];
        for (int d = 0; d < ndim; ++d) {
            out_shape[d] = input_shape[d];
        }
        out_shape[ndim - 1] *= 2;
        for (int d = ndim - 1; d >= 0; --d) {
            out_coords[d] = remaining % out_shape[d];
            remaining /= out_shape[d];
        }
        int64_t in_coords[8];
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

extern "C" void custom_transpose_launcher(const float* input, float* output,
                                           const int64_t* input_shape, const int64_t* input_strides,
                                           int ndim, int dim, size_t total_elements,
                                           cudaStream_t stream) {
    int64_t* d_shape;
    int64_t* d_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpyAsync(d_shape, input_shape, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_strides, input_strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    transpose_kernel<<<blocks, threads, 0, stream>>>(input, output, d_shape, d_strides, ndim, dim, total_elements);

    cudaFree(d_shape);
    cudaFree(d_strides);
}

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
