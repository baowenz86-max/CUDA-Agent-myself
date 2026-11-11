#include <cuda_runtime.h>
#include <math.h>

__global__ void gelu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t numel) {
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < numel;
         idx += stride) {
        float x = input[idx];
        // PyTorch default gelu: 0.5 * x * (1 + erf(x / sqrt(2)))
        output[idx] = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
    }
}

__global__ void hardswish_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int64_t numel) {
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < numel;
         idx += stride) {
        float x = input[idx];
        // hardswish: x * clamp(x + 3, 0, 6) / 6
        float clamped = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        output[idx] = x * clamped * (1.0f / 6.0f);
    }
}

extern "C" void gelu_forward_launcher(const float* input,
                                       float* output,
                                       int64_t numel,
                                       cudaStream_t stream) {
    const int block_size = 256;
    int64_t blocks64 = (numel + block_size - 1) / block_size;
    int blocks = (int)min(blocks64, (int64_t)2147483647);
    gelu_kernel<<<blocks, block_size, 0, stream>>>(input, output, numel);
}

extern "C" void hardswish_forward_launcher(const float* input,
                                           float* output,
                                           int64_t numel,
                                           cudaStream_t stream) {
    const int block_size = 256;
    int64_t blocks64 = (numel + block_size - 1) / block_size;
    int blocks = (int)min(blocks64, (int64_t)2147483647);
    hardswish_kernel<<<blocks, block_size, 0, stream>>>(input, output, numel);
}
