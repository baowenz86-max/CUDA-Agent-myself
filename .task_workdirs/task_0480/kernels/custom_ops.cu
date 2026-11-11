#include <cuda_runtime.h>
#include <math.h>

// Elementwise ReLU: y = max(0, x)
__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        output[i] = fmaxf(input[i], 0.0f);
    }
}

// Elementwise softplus with numerical stability: y = log1p(exp(x)), for x > threshold use x
__global__ void softplus_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float x = input[i];
        // PyTorch default beta=1, threshold=20
        if (x > 20.0f) {
            output[i] = x;
        } else {
            output[i] = log1pf(expf(x));
        }
    }
}

// Launcher for relu_forward
extern "C" void relu_forward_launcher(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid-stride loop handles rest
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

// Launcher for softplus_forward
extern "C" void softplus_forward_launcher(const float* input, float* output, int64_t n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    softplus_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}
