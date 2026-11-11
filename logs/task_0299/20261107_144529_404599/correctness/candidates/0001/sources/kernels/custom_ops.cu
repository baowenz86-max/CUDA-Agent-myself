#include <cuda_runtime.h>
#include <cmath>

// Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
__global__ void gelu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t total_elements,
    int ndim,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ output_strides)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t linear_idx = idx; linear_idx < total_elements; linear_idx += stride) {
        // 将线性索引转换为多维坐标
        int64_t offset_in = 0;
        int64_t offset_out = 0;
        int64_t remaining = linear_idx;
        for (int dim = ndim - 1; dim >= 0; --dim) {
            int64_t coord = remaining % sizes[dim];
            remaining /= sizes[dim];
            offset_in += coord * input_strides[dim];
            offset_out += coord * output_strides[dim];
        }

        float x = input[offset_in];
        float y = 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
        output[offset_out] = y;
    }
}

// SELU: scale * (max(x,0) + min(0, alpha * (exp(x) - 1)))
__global__ void selu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t total_elements,
    int ndim,
    const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ output_strides)
{
    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t linear_idx = idx; linear_idx < total_elements; linear_idx += stride) {
        int64_t offset_in = 0;
        int64_t offset_out = 0;
        int64_t remaining = linear_idx;
        for (int dim = ndim - 1; dim >= 0; --dim) {
            int64_t coord = remaining % sizes[dim];
            remaining /= sizes[dim];
            offset_in += coord * input_strides[dim];
            offset_out += coord * output_strides[dim];
        }

        float x = input[offset_in];
        float val = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
        output[offset_out] = scale * val;
    }
}

extern "C" void gelu_forward_launcher(
    float* output,
    const float* input,
    int64_t total_elements,
    int ndim,
    const int64_t* sizes,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream)
{
    if (total_elements == 0) return;

    const int block_size = 256;
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535; // grid-stride loop will handle rest

    gelu_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, total_elements, ndim, sizes, input_strides, output_strides);
}

extern "C" void selu_forward_launcher(
    float* output,
    const float* input,
    int64_t total_elements,
    int ndim,
    const int64_t* sizes,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream)
{
    if (total_elements == 0) return;

    const int block_size = 256;
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535;

    selu_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, total_elements, ndim, sizes, input_strides, output_strides);
}
