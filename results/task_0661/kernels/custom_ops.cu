#include <cuda_runtime.h>
#include <cstdint>

// Kernel to compute mean over a specified dimension (dim) of a tensor.
// Input: x (contiguous float tensor), shape array, ndim.
// Output: out (contiguous float tensor with reduced dimension removed).
// This kernel is launched with one thread per output element; each thread
// sums over the reduced dimension size and divides by that size.
// The indexing uses strides for the original tensor.
__global__ void mean_dim_kernel(const float* __restrict__ x,
                                float* __restrict__ out,
                                const int64_t* __restrict__ shape,
                                const int64_t* __restrict__ stride,
                                int64_t ndim,
                                int64_t reduce_dim,
                                int64_t reduce_size) {
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_out = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != reduce_dim) total_out *= shape[d];
    }
    if (out_idx >= total_out) return;

    // Convert linear output index to multi-dimensional index excluding reduce_dim.
    int64_t remaining = out_idx;
    int64_t in_index = 0;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        if (d == reduce_dim) continue;
        int64_t dim_size = shape[d];
        int64_t coord = remaining % dim_size;
        remaining /= dim_size;
        in_index += coord * stride[d];
    }

    float sum = 0.0f;
    for (int64_t r = 0; r < reduce_size; ++r) {
        sum += x[in_index + r * stride[reduce_dim]];
    }
    out[out_idx] = sum / static_cast<float>(reduce_size);
}

// Kernel to check elementwise equality between x and expanded mean.
// Each thread handles multiple elements in a grid-stride loop.
// If any element is unequal, it sets a device flag to 0 using atomicExch.
// The flag is initialized to 1 by the launcher via cudaMemset.
__global__ void equal_expand_as_kernel(const float* __restrict__ x,
                                       const float* __restrict__ mean,
                                       int64_t batch_size,
                                       int64_t spatial_size,
                                       int* __restrict__ equal_flag) {
    int64_t total = batch_size * spatial_size;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        int64_t s = i % spatial_size;
        if (x[i] != mean[s]) {
            atomicExch(equal_flag, 0);
            return;
        }
    }
}

// Launcher for mean over dimension
// Parameters: x pointer, out pointer, dim (0-based), shape array (length ndim), ndim, stream.
// Assume x is contiguous. We allocate shape/stride on device with cudaMalloc,
// copy them, launch kernel, and free after stream synchronization.
extern "C" void mean_dim_launcher(const float* x, float* out, int64_t dim, const int64_t* shape, int64_t ndim, cudaStream_t stream) {
    // Compute strides from shape (contiguous assumption)
    int64_t stride[8];
    int64_t prod = 1;
    for (int64_t d = ndim - 1; d >= 0; --d) {
        stride[d] = prod;
        prod *= shape[d];
    }
    int64_t reduce_size = shape[dim];

    int64_t* d_shape;
    int64_t* d_stride;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_stride, ndim * sizeof(int64_t));
    cudaMemcpyAsync(d_shape, shape, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_stride, stride, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    int64_t total_out = 1;
    for (int64_t d = 0; d < ndim; ++d) {
        if (d != dim) total_out *= shape[d];
    }
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    mean_dim_kernel<<<blocks, threads, 0, stream>>>(x, out, d_shape, d_stride, ndim, dim, reduce_size);

    // Synchronize to ensure kernel completion before freeing device memory.
    cudaStreamSynchronize(stream);
    cudaFree(d_shape);
    cudaFree(d_stride);
}

// Launcher for equal_expand_as
// Parameters: x, mean, batch_size, spatial_size, equal_out (host int pointer), stream.
// Allocates a device flag, initializes to 1, launches a single kernel that sets it to 0 if any mismatch.
// Then copies device flag to host and frees device memory after stream sync.
extern "C" void equal_expand_as_launcher(const float* x, const float* mean, int64_t batch_size, int64_t spatial_size, int* equal_out, cudaStream_t stream) {
    int64_t total = batch_size * spatial_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Limit blocks to avoid excessive resource usage; grid-stride handles the rest.
    if (blocks > 1024) blocks = 1024;

    int* d_equal_flag;
    cudaMalloc(&d_equal_flag, sizeof(int));
    cudaMemsetAsync(d_equal_flag, 1, sizeof(int), stream);

    equal_expand_as_kernel<<<blocks, threads, 0, stream>>>(x, mean, batch_size, spatial_size, d_equal_flag);

    cudaMemcpyAsync(equal_out, d_equal_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_equal_flag);
}
