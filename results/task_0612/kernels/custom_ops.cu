#include <cuda_runtime.h>
#include <cstdint>

// Specialized kernel for task: sum over dim=1 of contiguous 4D [B,C,H,W] -> [B,H,W]
__global__ void custom_sum_dim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t total_output)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    int64_t HW = H * W;
    int64_t CHW = C * HW;

    for (int64_t i = idx; i < total_output; i += stride) {
        // Decode output index (b, h, w)
        int64_t w = i % W;
        int64_t tmp = i / W;
        int64_t h = tmp % H;
        int64_t b = tmp / H;

        // Base input offset: b*CHW + h*W + w (assuming C is innermost after b? Actually input layout [B,C,H,W], so b*CHW + c*HW + h*W + w)
        int64_t base = b * CHW + h * W + w;
        float sum = 0.0f;
        for (int64_t c = 0; c < C; ++c) {
            sum += input[base + c * HW];
        }
        output[i] = sum;
    }
}

extern "C" void custom_sum_dim_launcher(
    const float* input,
    float* output,
    int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t total_output,
    cudaStream_t stream)
{
    int block_size = 256;
    int64_t grid_size = (total_output + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL;
    custom_sum_dim_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input, output, B, C, H, W, total_output);
}

__global__ void custom_expm1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        output[i] = expm1f(input[i]);
    }
}

extern "C" void custom_expm1_launcher(
    const float* input,
    float* output,
    int64_t total,
    cudaStream_t stream)
{
    int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL;
    custom_expm1_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input, output, total);
}
