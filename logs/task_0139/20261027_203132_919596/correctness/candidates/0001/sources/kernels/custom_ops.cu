#include <cuda_runtime.h>

// Leaky ReLU kernel: output = input if input >= 0 else negative_slope * input
__global__ void leaky_relu_kernel(const float* input, float* output, int64_t numel, float negative_slope) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        float val = input[idx];
        output[idx] = val >= 0.0f ? val : negative_slope * val;
    }
}

// SELU kernel: output = scale * (max(0, input) + min(0, alpha * (exp(input) - 1)))
__global__ void selu_kernel(const float* input, float* output, int64_t numel) {
    const float alpha = 1.6732632423543772848170429916717f;
    const float scale = 1.0507009873554804934193349852946f;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        float val = input[idx];
        float result;
        if (val > 0.0f) {
            result = scale * val;
        } else {
            result = scale * (alpha * (expf(val) - 1.0f));
        }
        output[idx] = result;
    }
}

// Launcher for leaky_relu_forward. Assumes input and output are contiguous float32 arrays.
extern "C" void leaky_relu_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    float negative_slope,
    cudaStream_t stream) {
    
    const int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) {
        blocks = 65535;
    }
    leaky_relu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel, negative_slope);
}

// Launcher for selu_forward. Assumes input and output are contiguous float32 arrays.
extern "C" void selu_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream) {
    
    const int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) {
        blocks = 65535;
    }
    selu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}
