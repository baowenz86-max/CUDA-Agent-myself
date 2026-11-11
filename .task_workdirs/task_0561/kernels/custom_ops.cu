#include <cuda_runtime.h>
#include <cmath>

// Kernel: replace NaN with zero
extern "C" __global__ void custom_nan_to_zero_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = isnan(input[i]) ? 0.0f : input[i];
    }
}

// Kernel: PReLU with per-channel weight
extern "C" __global__ void custom_prelu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    int64_t numel,
    int64_t channels,
    int64_t hw)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        int64_t c = (i / hw) % channels;
        float val = input[i];
        output[i] = (val > 0.0f) ? val : (weight[c] * val);
    }
}

// Launcher for custom_nan_to_zero
extern "C" void custom_nan_to_zero_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream)
{
    const int blockSize = 256;
    int64_t gridSize = (numel + blockSize - 1) / blockSize;
    if (gridSize > 0) {
        custom_nan_to_zero_kernel<<<(unsigned int)gridSize, blockSize, 0, stream>>>(
            output, input, numel);
    }
}

// Launcher for custom_prelu
extern "C" void custom_prelu_launcher(
    float* output,
    const float* input,
    const float* weight,
    int64_t numel,
    int64_t channels,
    int64_t hw,
    cudaStream_t stream)
{
    const int blockSize = 256;
    int64_t gridSize = (numel + blockSize - 1) / blockSize;
    if (gridSize > 0) {
        custom_prelu_kernel<<<(unsigned int)gridSize, blockSize, 0, stream>>>(
            output, input, weight, numel, channels, hw);
    }
}
