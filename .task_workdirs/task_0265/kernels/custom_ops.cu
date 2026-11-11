#include <cuda_runtime.h>

// Fused addcdiv: out = x + value * (tensor1 / tensor2)
// All tensors have same shape and float32.
__global__ void fma_addcdiv_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ tensor1,
    const float* __restrict__ tensor2,
    float value,
    int total_elements,
    int dim0, int dim1, int dim2, int dim3,
    int stride_x0, int stride_x1, int stride_x2, int stride_x3,
    int stride_t1_0, int stride_t1_1, int stride_t1_2, int stride_t1_3,
    int stride_t2_0, int stride_t2_1, int stride_t2_2, int stride_t2_3)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_elements; i += stride) {
        // Decompose linear index into 4D coordinates
        int i0 = i / (dim1 * dim2 * dim3);
        int rem = i % (dim1 * dim2 * dim3);
        int i1 = rem / (dim2 * dim3);
        rem = rem % (dim2 * dim3);
        int i2 = rem / dim3;
        int i3 = rem % dim3;

        // Compute element offsets using strides
        int x_idx = i0 * stride_x0 + i1 * stride_x1 + i2 * stride_x2 + i3 * stride_x3;
        int t1_idx = i0 * stride_t1_0 + i1 * stride_t1_1 + i2 * stride_t1_2 + i3 * stride_t1_3;
        int t2_idx = i0 * stride_t2_0 + i1 * stride_t2_1 + i2 * stride_t2_2 + i3 * stride_t2_3;

        out[i] = x[x_idx] + value * (tensor1[t1_idx] / tensor2[t2_idx]);
    }
}

// Add bias with broadcasting along dimension 1: out = x + bias[i1]
__global__ void add_bias_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ bias,
    int total_elements,
    int dim0, int dim1, int dim2, int dim3,
    int stride_x0, int stride_x1, int stride_x2, int stride_x3)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_elements; i += stride) {
        // Decompose linear index
        int i0 = i / (dim1 * dim2 * dim3);
        int rem = i % (dim1 * dim2 * dim3);
        int i1 = rem / (dim2 * dim3);
        rem = rem % (dim2 * dim3);
        int i2 = rem / dim3;
        int i3 = rem % dim3;

        int x_idx = i0 * stride_x0 + i1 * stride_x1 + i2 * stride_x2 + i3 * stride_x3;
        // bias is contiguous with shape [dim1,1,1], stride (1,1,1)
        out[i] = x[x_idx] + bias[i1];
    }
}

extern "C" void fma_addcdiv_launcher(
    float* out,
    const float* x,
    const float* tensor1,
    const float* tensor2,
    float value,
    int total_elements,
    int dim0, int dim1, int dim2, int dim3,
    int stride_x0, int stride_x1, int stride_x2, int stride_x3,
    int stride_t1_0, int stride_t1_1, int stride_t1_2, int stride_t1_3,
    int stride_t2_0, int stride_t2_1, int stride_t2_2, int stride_t2_3,
    cudaStream_t stream)
{
    const int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    // Cap grid size at 65535; grid-stride loop handles larger
    if (grid_size > 65535) grid_size = 65535;

    fma_addcdiv_kernel<<<grid_size, block_size, 0, stream>>>(
        out, x, tensor1, tensor2, value,
        total_elements,
        dim0, dim1, dim2, dim3,
        stride_x0, stride_x1, stride_x2, stride_x3,
        stride_t1_0, stride_t1_1, stride_t1_2, stride_t1_3,
        stride_t2_0, stride_t2_1, stride_t2_2, stride_t2_3);
}

extern "C" void add_bias_launcher(
    float* out,
    const float* x,
    const float* bias,
    int total_elements,
    int dim0, int dim1, int dim2, int dim3,
    int stride_x0, int stride_x1, int stride_x2, int stride_x3,
    cudaStream_t stream)
{
    const int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;

    add_bias_kernel<<<grid_size, block_size, 0, stream>>>(
        out, x, bias,
        total_elements,
        dim0, dim1, dim2, dim3,
        stride_x0, stride_x1, stride_x2, stride_x3);
}
