#include <cuda_runtime.h>
#include <cstdint>

// ReLU forward kernel: y = max(0, x)
// Supports arbitrary strides via linear index and manual strided access.
__global__ void relu_forward_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    int64_t numel,
    int ndim,
    const int64_t* __restrict__ shape,
    const int64_t* __restrict__ x_strides,
    const int64_t* __restrict__ y_strides)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < numel; linear += stride) {
        int64_t remaining = linear;
        int64_t x_index = 0;
        int64_t y_index = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            int64_t coord = remaining % shape[d];
            remaining /= shape[d];
            x_index += coord * x_strides[d];
            y_index += coord * y_strides[d];
        }
        y[y_index] = fmaxf(x[x_index], 0.0f);
    }
}

extern "C" void relu_forward_launcher(
    float* y,
    const float* x,
    int64_t numel,
    int ndim,
    const int64_t* shape,
    const int64_t* x_strides,
    const int64_t* y_strides,
    cudaStream_t stream)
{
    if (numel == 0) return;

    // Copy metadata to device memory so kernel can access it.
    int64_t* d_shape;
    int64_t* d_x_strides;
    int64_t* d_y_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_x_strides, ndim * sizeof(int64_t));
    cudaMalloc(&d_y_strides, ndim * sizeof(int64_t));
    cudaMemcpyAsync(d_shape, shape, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_x_strides, x_strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_y_strides, y_strides, ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    const int block_size = 256;
    int grid_size = (int)((numel + block_size - 1) / block_size);
    if (grid_size > 65535) {
        grid_size = 65535;
    }
    relu_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        y, x, numel, ndim, d_shape, d_x_strides, d_y_strides);

    cudaFree(d_shape);
    cudaFree(d_x_strides);
    cudaFree(d_y_strides);
}

// Tensor equal kernel: compares two tensors elementwise, writes mismatch flag to device memory.
__global__ void tensor_equal_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    int64_t numel,
    int* __restrict__ flag)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < numel; linear += stride) {
        if (a[linear] != b[linear]) {
            atomicExch(flag, 0);
        }
    }
}

extern "C" bool tensor_equal_launcher(
    const float* a,
    const float* b,
    int64_t numel,
    cudaStream_t stream)
{
    if (numel == 0) return true;
    int* d_flag;
    cudaMalloc(&d_flag, sizeof(int));
    int init_val = 1;
    cudaMemcpyAsync(d_flag, &init_val, sizeof(int), cudaMemcpyHostToDevice, stream);
    const int block_size = 256;
    int grid_size = (int)((numel + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535;
    tensor_equal_kernel<<<grid_size, block_size, 0, stream>>>(a, b, numel, d_flag);
    int h_flag;
    cudaMemcpyAsync(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_flag);
    return h_flag == 1;
}
