#include <cuda_runtime.h>
#include <cmath>

// Elementwise ReLU: out[i] = max(in[i], 0)
__global__ void relu_kernel(const float* __restrict__ in, float* __restrict__ out, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < n; i += stride) {
        float v = in[i];
        out[i] = v > 0.0f ? v : 0.0f;
    }
}

// Exact GELU: out[i] = 0.5 * x * (1 + erf(x / sqrt(2)))
__global__ void gelu_kernel(const float* __restrict__ in, float* __restrict__ out, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    const float inv_sqrt2 = 0.70710678118654752440f;
    for (long long i = idx; i < n; i += stride) {
        float v = in[i];
        out[i] = 0.5f * v * (1.0f + erff(v * inv_sqrt2));
    }
}

// Launcher for ReLU. Input and output are contiguous float32 tensors on current CUDA stream.
extern "C" void custom_relu_launcher(const float* input, float* output, long long numel, cudaStream_t stream) {
    const int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

// Launcher for exact GELU. Input and output are contiguous float32 tensors on current CUDA stream.
extern "C" void custom_gelu_launcher(const float* input, float* output, long long numel, cudaStream_t stream) {
    const int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;
    gelu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}
