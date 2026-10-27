#include <cuda_runtime.h>

__global__ void exp_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        output[i] = expf(input[i]);
    }
}

extern "C" void exp_forward_launcher(const float* input, float* output, int n, cudaStream_t stream) {
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks == 0) return;
    exp_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // error handling left to binding
    }
}
