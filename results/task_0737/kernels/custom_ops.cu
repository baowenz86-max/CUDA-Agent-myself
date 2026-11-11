#include <cuda_runtime.h>

// Element-wise ReLU: out = max(x, 0)
__global__ void relu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t n) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float val = input[i];
        output[i] = (val > 0.0f) ? val : 0.0f;
    }
}

// Launcher used by binding.cpp
extern "C" void relu_forward_launcher(const float* input,
                                     float* output,
                                     int64_t n,
                                     cudaStream_t stream) {
    if (n <= 0) return;
    const int block_size = 256;
    // Cap grid to avoid exceeding limits for huge tensors; grid-stride loop handles rest.
    // Maximum grid dimension in x is 2^31 - 1, so no need to cap for realistic n.
    int64_t grid_size = (n + block_size - 1) / block_size;
    if (grid_size > 1073741824LL) grid_size = 1073741824LL; // safety cap
    relu_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(input, output, n);
}
