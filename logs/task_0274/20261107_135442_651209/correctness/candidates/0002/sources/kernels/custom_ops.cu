#include <cuda_runtime.h>

// Simple elementwise ReLU kernel
__global__ void relu_kernel(const float* input, float* output, long long numel) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = val > 0.0f ? val : 0.0f;
    }
}

// Simple elementwise absolute value kernel
__global__ void abs_kernel(const float* input, float* output, long long numel) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = val < 0.0f ? -val : val;
    }
}

extern "C" void relu_forward_launcher(const float* input, float* output, long long numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid cap, grid-stride loop handles rest
    if (blocks == 0) return;
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void abs_forward_launcher(const float* input, float* output, long long numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) return;
    abs_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}
