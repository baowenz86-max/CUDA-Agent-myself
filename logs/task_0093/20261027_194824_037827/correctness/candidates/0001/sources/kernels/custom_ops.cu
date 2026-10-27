#include <cuda_runtime.h>

// CUDA kernel for mul_bias_forward
// input: [N, C, L] float32 contiguous
// bias: [C, 1] float32 contiguous
// output: [N, C, L] float32 contiguous
// For each element: output[n, c, l] = input[n, c, l] * bias[c]
__global__ void mul_bias_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int L,
    long long total_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < total_elements; i += stride) {
        int l = i % L;
        int c = (i / L) % C;
        output[i] = input[i] * bias[c];
    }
}

// Launcher for mul_bias_forward
extern "C" void mul_bias_forward_launcher(
    const float* input,
    const float* bias,
    float* output,
    int N, int C, int L,
    cudaStream_t stream) {
    long long total = (long long)N * C * L;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    if (blocks > 2147483647LL) blocks = 2147483647LL; // max grid dim
    if (blocks == 0) return;
    mul_bias_forward_kernel<<<(int)blocks, threads, 0, stream>>>(
        input, bias, output, N, C, L, total);
}
