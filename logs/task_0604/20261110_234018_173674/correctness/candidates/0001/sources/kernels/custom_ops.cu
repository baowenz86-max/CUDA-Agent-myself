#include <cuda_runtime.h>

// Helper to compute linear offset from multi-dimensional index given strides
__device__ __forceinline__ int64_t compute_offset(
    int64_t flat_idx,
    const int64_t* strides,
    const int64_t* dims,
    int ndims)
{
    int64_t offset = 0;
    int64_t idx = flat_idx;
    for (int d = ndims - 1; d >= 0; --d) {
        int64_t coord = idx % dims[d];
        idx /= dims[d];
        offset += coord * strides[d];
    }
    return offset;
}

// Element-wise absolute value kernel, stride-aware
__global__ void abs_kernel(
    const float* input,
    float* output,
    int64_t total_elements,
    const int64_t* input_strides,
    const int64_t* output_strides,
    const int64_t* dims,
    int ndims)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t in_offset = compute_offset(i, input_strides, dims, ndims);
        int64_t out_offset = compute_offset(i, output_strides, dims, ndims);
        output[out_offset] = fabsf(input[in_offset]);
    }
}

// Element-wise ReLU6 kernel, stride-aware
__global__ void relu6_kernel(
    const float* input,
    float* output,
    int64_t total_elements,
    const int64_t* input_strides,
    const int64_t* output_strides,
    const int64_t* dims,
    int ndims)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t in_offset = compute_offset(i, input_strides, dims, ndims);
        int64_t out_offset = compute_offset(i, output_strides, dims, ndims);
        float v = input[in_offset];
        v = fminf(fmaxf(v, 0.0f), 6.0f);
        output[out_offset] = v;
    }
}

// Fused abs then ReLU6 kernel, stride-aware
__global__ void fused_abs_relu6_kernel(
    const float* input,
    float* output,
    int64_t total_elements,
    const int64_t* input_strides,
    const int64_t* output_strides,
    const int64_t* dims,
    int ndims)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t in_offset = compute_offset(i, input_strides, dims, ndims);
        int64_t out_offset = compute_offset(i, output_strides, dims, ndims);
        float v = fabsf(input[in_offset]);
        v = fminf(v, 6.0f);
        output[out_offset] = v;
    }
}

// Launch configuration
extern "C" void abs_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    const int64_t* input_strides,
    const int64_t* output_strides,
    const int64_t* dims,
    int ndims,
    cudaStream_t stream)
{
    if (total_elements <= 0) return;
    const int blockSize = 256;
    int64_t blocks = (total_elements + blockSize - 1) / blockSize;
    // Cap grid size to avoid exceeding limits, grid-stride loop handles rest
    const int64_t maxGridSize = 2147483647; // 2^31 - 1
    blocks = (blocks > maxGridSize) ? maxGridSize : blocks;
    abs_kernel<<<(unsigned int)blocks, blockSize, 0, stream>>>(
        input, output, total_elements, input_strides, output_strides, dims, ndims);
}

extern "C" void relu6_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    const int64_t* input_strides,
    const int64_t* output_strides,
    const int64_t* dims,
    int ndims,
    cudaStream_t stream)
{
    if (total_elements <= 0) return;
    const int blockSize = 256;
    int64_t blocks = (total_elements + blockSize - 1) / blockSize;
    const int64_t maxGridSize = 2147483647;
    blocks = (blocks > maxGridSize) ? maxGridSize : blocks;
    relu6_kernel<<<(unsigned int)blocks, blockSize, 0, stream>>>(
        input, output, total_elements, input_strides, output_strides, dims, ndims);
}

extern "C" void abs_relu6_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    const int64_t* input_strides,
    const int64_t* output_strides,
    const int64_t* dims,
    int ndims,
    cudaStream_t stream)
{
    if (total_elements <= 0) return;
    const int blockSize = 256;
    int64_t blocks = (total_elements + blockSize - 1) / blockSize;
    const int64_t maxGridSize = 2147483647;
    blocks = (blocks > maxGridSize) ? maxGridSize : blocks;
    fused_abs_relu6_kernel<<<(unsigned int)blocks, blockSize, 0, stream>>>(
        input, output, total_elements, input_strides, output_strides, dims, ndims);
}
