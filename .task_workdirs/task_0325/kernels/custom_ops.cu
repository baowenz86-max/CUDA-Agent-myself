#include <cuda_runtime.h>
#include <math.h>

__global__ void softplus_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        if (x <= 20.0f) {
            output[i] = log1pf(expf(x));
        } else {
            output[i] = x;
        }
    }
}

__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        output[i] = fmaxf(0.0f, x);
    }
}

extern "C" void softplus_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) return;
    softplus_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void relu_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) return;
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}
