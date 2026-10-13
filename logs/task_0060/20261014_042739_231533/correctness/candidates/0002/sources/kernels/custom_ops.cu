// kernels/custom_ops.cu
#include <cuda_runtime.h>

__global__ void fused_relu_hardsigmoid_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long numel)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < numel; idx += stride) {
        float val = input[idx];
        val = val > 0.0f ? val : 0.0f;  // relu
        val = val + 3.0f;
        val = val < 0.0f ? 0.0f : (val > 6.0f ? 6.0f : val); // clamp
        output[idx] = val * (1.0f / 6.0f);
    }
}

extern "C" void fused_relu_hardsigmoid_launcher(
    float* output,
    const float* input,
    long long numel,
    cudaStream_t stream)
{
    int block_size = 256;
    long long grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) {
        grid_size = 65535;
    }

    fused_relu_hardsigmoid_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        output, input, numel);
}