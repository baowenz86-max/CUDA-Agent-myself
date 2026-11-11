#include <cuda_runtime.h>

// ELU kernel: f(x) = x if x > 0 else alpha * (expf(x) - 1.f)
__global__ void elu_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           long long n,
                           float alpha) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < n; i += stride) {
        float x = input[i];
        output[i] = (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
    }
}

// ReLU kernel: f(x) = max(x, 0)
__global__ void relu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < n; i += stride) {
        float x = input[i];
        output[i] = (x > 0.0f) ? x : 0.0f;
    }
}

// Launcher for ELU
extern "C" cudaError_t custom_elu_launcher(const float* input,
                                            float* output,
                                            long long n,
                                            double alpha,
                                            cudaStream_t stream) {
    if (n <= 0) return cudaSuccess;
    const int threads = 256;
    // 确保 grid 大小足够，同时避免整数溢出
    long long blocks_ll = (n + threads - 1) / threads;
    // 限制 grid 大小为 2^31-1，但通常不会超过（n ≈ 2e8，blocks ≈ 786k）
    int blocks = (int)blocks_ll;
    float alpha_f = (float)alpha;
    elu_kernel<<<blocks, threads, 0, stream>>>(input, output, n, alpha_f);
    return cudaGetLastError();
}

// Launcher for ReLU
extern "C" cudaError_t custom_relu_launcher(const float* input,
                                             float* output,
                                             long long n,
                                             cudaStream_t stream) {
    if (n <= 0) return cudaSuccess;
    const int threads = 256;
    long long blocks_ll = (n + threads - 1) / threads;
    int blocks = (int)blocks_ll;
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
    return cudaGetLastError();
}
