#include <cuda_runtime.h>

// Element-wise absolute value kernel, contiguous input/output assumed
__global__ void abs_kernel(
    const float* input,
    float* output,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

// Element-wise ReLU6 kernel, contiguous input/output assumed
__global__ void relu6_kernel(
    const float* input,
    float* output,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        float v = input[i];
        v = fminf(fmaxf(v, 0.0f), 6.0f);
        output[i] = v;
    }
}

// Fused abs then ReLU6 kernel, contiguous input/output assumed
__global__ void fused_abs_relu6_kernel(
    const float* input,
    float* output,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        float v = fabsf(input[i]);
        v = fminf(v, 6.0f);
        output[i] = v;
    }
}

// Launch configuration
__host__ __device__ int64_t get_grid_size(int64_t n, int block_size) {
    int64_t blocks = (n + block_size - 1) / block_size;
    // Cap to a safe grid size (e.g., 1e6) to avoid too many blocks; grid-stride loop handles rest
    const int64_t max_grid = 65535 * 32; // 2,097,120
    if (blocks > max_grid) {
        blocks = max_grid;
    }
    return blocks;
}

extern "C" void abs_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream)
{
    if (total_elements <= 0) return;
    const int blockSize = 256;
    int64_t blocks = get_grid_size(total_elements, blockSize);
    abs_kernel<<<(unsigned int)blocks, blockSize, 0, stream>>>(
        input, output, total_elements);
}

extern "C" void relu6_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream)
{
    if (total_elements <= 0) return;
    const int blockSize = 256;
    int64_t blocks = get_grid_size(total_elements, blockSize);
    relu6_kernel<<<(unsigned int)blocks, blockSize, 0, stream>>>(
        input, output, total_elements);
}

extern "C" void abs_relu6_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream)
{
    if (total_elements <= 0) return;
    const int blockSize = 256;
    int64_t blocks = get_grid_size(total_elements, blockSize);
    fused_abs_relu6_kernel<<<(unsigned int)blocks, blockSize, 0, stream>>>(
        input, output, total_elements);
}
