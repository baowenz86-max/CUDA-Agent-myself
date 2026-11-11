#include <cuda_runtime.h>
#include <cstdint>

// Fixed dimensions for this task: 4D tensors
#define NDIM 4

__global__ void transpose_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  const int64_t* __restrict__ input_shape,
                                  const int64_t* __restrict__ input_strides,
                                  const int64_t* __restrict__ output_strides,
                                  int dim,
                                  size_t total_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t linear_idx = idx; linear_idx < total_elements; linear_idx += stride) {
        // Decode linear_idx into logical output coordinates assuming contiguous enumeration
        // of the transposed shape (row-major on the logical view)
        size_t remaining = linear_idx;
        int64_t out_coords[NDIM];
        int64_t out_shape[NDIM];
        for (int i = 0; i < NDIM; ++i) {
            out_shape[i] = input_shape[i];
        }
        // swap dim and last to get output shape
        int64_t tmp = out_shape[dim];
        out_shape[dim] = out_shape[NDIM - 1];
        out_shape[NDIM - 1] = tmp;

        for (int d = NDIM - 1; d >= 0; --d) {
            out_coords[d] = remaining % out_shape[d];
            remaining /= out_shape[d];
        }

        // Compute output storage offset using output_strides
        size_t output_offset = 0;
        for (int d = 0; d < NDIM; ++d) {
            output_offset += out_coords[d] * output_strides[d];
        }

        // Compute input coordinates by swapping dim and last
        int64_t in_coords[NDIM];
        for (int d = 0; d < NDIM; ++d) {
            in_coords[d] = out_coords[d];
        }
        int64_t tmp_coord = in_coords[dim];
        in_coords[dim] = in_coords[NDIM - 1];
        in_coords[NDIM - 1] = tmp_coord;

        // Compute input offset
        size_t input_offset = 0;
        for (int d = 0; d < NDIM; ++d) {
            input_offset += in_coords[d] * input_strides[d];
        }

        output[output_offset] = input[input_offset];
    }
}

__global__ void cat2x_kernel(const float* __restrict__ input,
                              float* __restrict__ output,
                              const int64_t* __restrict__ input_shape,
                              const int64_t* __restrict__ input_strides,
                              size_t total_output_elements) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    int64_t last_dim_size = input_shape[NDIM - 1];
    for (size_t linear_idx = idx; linear_idx < total_output_elements; linear_idx += stride) {
        // Output is contiguous, decode linear_idx into output coordinates
        size_t remaining = linear_idx;
        int64_t out_coords[NDIM];
        int64_t out_shape[NDIM];
        for (int d = 0; d < NDIM; ++d) {
            out_shape[d] = input_shape[d];
        }
        out_shape[NDIM - 1] *= 2;
        for (int d = NDIM - 1; d >= 0; --d) {
            out_coords[d] = remaining % out_shape[d];
            remaining /= out_shape[d];
        }

        // Map to input coordinates: last dim mod last_dim_size
        int64_t in_coords[NDIM];
        for (int d = 0; d < NDIM; ++d) {
            in_coords[d] = out_coords[d];
        }
        in_coords[NDIM - 1] = out_coords[NDIM - 1] % last_dim_size;

        size_t input_offset = 0;
        for (int d = 0; d < NDIM; ++d) {
            input_offset += in_coords[d] * input_strides[d];
        }

        output[linear_idx] = input[input_offset];
    }
}

extern "C" void custom_transpose_launcher(const float* input, float* output,
                                           const int64_t* input_shape, const int64_t* input_strides,
                                           const int64_t* output_strides,
                                           int ndim, int dim, size_t total_elements,
                                           cudaStream_t stream) {
    // Since ndim is always 4 for this task, we can copy shape/stride into constant memory
    // or pass as kernel parameters via a struct. To avoid dynamic allocation, we use
    // __constant__ arrays or directly pass them as kernel arguments using cudaMemcpyToSymbol.
    // For simplicity, we use cudaMemcpyToSymbolAsync to constant memory.
    // But we can also pass pointers to device memory; however we don't have device memory.
    // Better: allocate device memory for shape/strides with cudaMalloc, copy, launch, then
    // cudaStreamSynchronize before free to ensure safety. Or use cudaFreeAsync.
    // We'll use cudaMalloc/cudaFreeAsync to avoid synchronization overhead.
    int64_t* d_shape;
    int64_t* d_strides;
    int64_t* d_out_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMalloc(&d_out_strides, ndim * sizeof(int64_t));
    cudaMemcpyAsync(d_shape, input_shape, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_strides, input_strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_out_strides, output_strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    transpose_kernel<<<blocks, threads, 0, stream>>>(input, output, d_shape, d_strides, d_out_strides, dim, total_elements);

    // Use cudaFreeAsync to safely free after kernel completion on the stream
    cudaFreeAsync(d_shape, stream);
    cudaFreeAsync(d_strides, stream);
    cudaFreeAsync(d_out_strides, stream);
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
    cat2x_kernel<<<blocks, threads, 0, stream>>>(input, output, d_shape, d_strides, total_output_elements);

    cudaFreeAsync(d_shape, stream);
    cudaFreeAsync(d_strides, stream);
}
