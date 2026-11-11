#include <cuda_runtime.h>
#include <cmath>

// Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
__global__ void gelu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t linear_idx = idx; linear_idx < total_elements; linear_idx += stride) {
        float x = input[linear_idx];
        float y = 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
        output[linear_idx] = y;
    }
}

// SELU: scale * (max(x,0) + min(0, alpha * (exp(x) - 1)))
__global__ void selu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t total_elements)
{
    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t linear_idx = idx; linear_idx < total_elements; linear_idx += stride) {
        float x = input[linear_idx];
        float val = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
        output[linear_idx] = scale * val;
    }
}

extern "C" void gelu_forward_launcher(
    float* output,
    const float* input,
    int64_t total_elements,
    cudaStream_t stream)
{
    if (total_elements == 0) return;

    const int block_size = 256;
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535; // grid-stride loop will handle rest

    gelu_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, total_elements);
}

extern "C" void selu_forward_launcher(
    float* output,
    const float* input,
    int64_t total_elements,
    cudaStream_t stream)
{
    if (total_elements == 0) return;

    const int block_size = 256;
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535;

    selu_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, total_elements);
}
