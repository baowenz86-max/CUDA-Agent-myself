#include <cuda_runtime.h>

// Elementwise clamp: output = min(max(x, 0.0f), 1.0f)
__global__ void clamp_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float v = input[i];
        v = fminf(fmaxf(v, 0.0f), 1.0f);
        output[i] = v;
    }
}

// Fused clamp + hardswish: y = clamp(x,0,1); out = y * (y + 3) / 6
__global__ void clamp_hardswish_kernel(const float* __restrict__ input,
                                       float* __restrict__ output,
                                       int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float v = input[i];
        v = fminf(fmaxf(v, 0.0f), 1.0f);
        output[i] = v * (v + 3.0f) * (1.0f / 6.0f);
    }
}

static inline int compute_blocks(int64_t n, int block_size) {
    int64_t blocks64 = (n + block_size - 1) / block_size;
    // Cap at maximum grid dimension (2^31-1) for safety, but typical n won't hit it.
    int max_blocks = 2147483647;
    return (int)(blocks64 < max_blocks ? blocks64 : max_blocks);
}

extern "C" void clamp_forward_launcher(const float* input,
                                       float* output,
                                       int64_t n,
                                       cudaStream_t stream) {
    const int block_size = 256;
    int blocks = compute_blocks(n, block_size);
    clamp_kernel<<<blocks, block_size, 0, stream>>>(input, output, n);
}

extern "C" void clamp_hardswish_forward_launcher(const float* input,
                                                 float* output,
                                                 int64_t n,
                                                 cudaStream_t stream) {
    const int block_size = 256;
    int blocks = compute_blocks(n, block_size);
    clamp_hardswish_kernel<<<blocks, block_size, 0, stream>>>(input, output, n);
}
