#include <cuda_runtime.h>

// Elementwise multiplication kernel for two tensors with identical shape.
// Supports arbitrary strides (non-contiguous) by computing per-dimension indices.
// Grid-stride loop ensures all elements are covered even if grid is smaller than data.
__global__ void elementwise_mul_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ y,
    int ndim,
    const int64_t* __restrict__ shape,
    const int64_t* __restrict__ stride_out,
    const int64_t* __restrict__ stride_x,
    const int64_t* __restrict__ stride_y,
    int64_t numel)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t step = (int64_t)blockDim.x * gridDim.x;

    for (; idx < numel; idx += step) {
        // Convert linear index to multi-dimensional index
        int64_t remaining = idx;
        int64_t offset_out = 0;
        int64_t offset_x = 0;
        int64_t offset_y = 0;

        // Dimension order: assume row-major, strides given in elements.
        for (int d = 0; d < ndim; ++d) {
            int64_t dim_size = shape[d];
            int64_t coord = remaining % dim_size;
            remaining /= dim_size;
            offset_out += coord * stride_out[d];
            offset_x += coord * stride_x[d];
            offset_y += coord * stride_y[d];
        }

        out[offset_out] = x[offset_x] * y[offset_y];
    }
}

// Launcher for mul_forward. All pointers are element offsets (float*).
// shape, stride_out, stride_x, stride_y are arrays of length ndim (int64_t).
// ndim is typically 4 for the provided tensor shape.
// stream is the current CUDA stream.
extern "C" void mul_forward_launcher(
    float* out,
    const float* x,
    const float* y,
    int ndim,
    const int64_t* shape,
    const int64_t* stride_out,
    const int64_t* stride_x,
    const int64_t* stride_y,
    int64_t numel,
    cudaStream_t stream)
{
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    // Limit grid size to a reasonable maximum (e.g., 2^31-1) to avoid launch failure.
    if (grid_size > 2147483647LL) {
        grid_size = 2147483647LL;
    }

    elementwise_mul_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        out, x, y, ndim, shape, stride_out, stride_x, stride_y, numel);
}
