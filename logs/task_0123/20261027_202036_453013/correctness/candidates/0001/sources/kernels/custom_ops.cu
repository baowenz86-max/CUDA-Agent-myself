#include <cuda_runtime.h>

// 逐元素指数 kernel
__global__ void exp_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        output[i] = expf(input[i]);
    }
}

// launcher: exp_forward
// input: 输入指针 (float)
// output: 输出指针 (float)
// n: 元素个数
// stream: CUDA stream
extern "C" void exp_forward_launcher(const float* input, float* output, int n, cudaStream_t stream) {
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks == 0) return;  // 空张量
    exp_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // 错误处理可留空，后续 binding 层检测
    }
}
