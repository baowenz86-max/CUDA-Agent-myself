#include <cuda_runtime.h>

// Fused ReLU + clamp: out = min(max(x, 0.0f), 1.0f)
__global__ void relu_clamp_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int64_t n) {
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < n;
         idx += stride) {
        float v = input[idx];
        v = fmaxf(v, 0.0f);
        v = fminf(v, 1.0f);
        output[idx] = v;
    }
}

// C interface for binding layer
extern "C" void relu_clamp_forward_launcher(const float* input,
                                            float* output,
                                            int64_t n,
                                            cudaStream_t stream) {
    if (n <= 0) return;
    const int threads = 256;
    const int64_t max_blocks_per_grid = 2147483647LL;
    int64_t blocks_64 = (n + threads - 1) / threads;
    if (blocks_64 > max_blocks_per_grid) {
        blocks_64 = max_blocks_per_grid;
    }
    int blocks = static_cast<int>(blocks_64);
    relu_clamp_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}
