#include <cuda_runtime.h>
#include <math.h>

// Elementwise Mish activation: y = x * tanh(softplus(x))
// softplus(x) = log(1 + exp(x)), computed stably as log1p(exp(-abs(x))) + max(x,0)
__global__ void mish_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        float sp = (x >= 0.0f) ? x + log1pf(expf(-x)) : log1pf(expf(x));
        output[i] = x * tanhf(sp);
    }
}

// C-interface launcher for mish_forward. Assumes input and output are contiguous float32 CUDA tensors with same numel.
extern "C" void mish_forward_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      cudaStream_t stream) {
    if (numel <= 0) return;
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    // Grid cap (2^31-1) is far above needed for this size, but keep grid-stride loop anyway for safety.
    if (blocks > 2147483647) blocks = 2147483647;
    mish_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}
