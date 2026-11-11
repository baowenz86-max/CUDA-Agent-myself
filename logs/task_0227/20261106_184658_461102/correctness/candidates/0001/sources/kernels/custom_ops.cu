#include <cuda_runtime.h>

// Hardshrink kernel: y = x if |x| > lambda else 0, lambda=0.5
__global__ void hardshrink_kernel(float* output, const float* input, int64_t N) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < N; i += stride) {
        float x = input[i];
        output[i] = (x > 0.5f || x < -0.5f) ? x : 0.0f;
    }
}

// Host launcher for binding layer
extern "C" void hardshrink_launcher(
    float* output,
    const float* input,
    int64_t N,
    cudaStream_t stream
) {
    if (N <= 0) return;
    const int threads = 256;
    int64_t blocks64 = (N + threads - 1) / threads;
    // Cap grid to avoid launch failure, use grid-stride loop above
    if (blocks64 > 2147483647LL) blocks64 = 2147483647LL;
    int blocks = (int)blocks64;
    hardshrink_kernel<<<blocks, threads, 0, stream>>>(output, input, N);
    cudaGetLastError(); // binding layer may also check, but this surfaces launch errors
}
