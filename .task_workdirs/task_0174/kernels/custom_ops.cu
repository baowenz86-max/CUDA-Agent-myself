#include <cuda_runtime.h>

// Elementwise hardtanh: y = clamp(x, -1.0f, 1.0f)
// Supports arbitrary strides: only numel matters.
__global__ void hardtanh_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t numel) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float v = input[i];
        v = fminf(fmaxf(v, -1.0f), 1.0f);
        output[i] = v;
    }
}

// C interface launcher (no PyTorch dependencies)
extern "C" void hardtanh_forward_launcher(float* output,
                                          const float* input,
                                          int64_t numel,
                                          cudaStream_t stream) {
    const int block = 256;
    // Cap grid to avoid exceeding maximum grid dimensions, use grid-stride loop.
    int64_t grid64 = (numel + block - 1) / block;
    if (grid64 > 1073741824LL) grid64 = 1073741824LL; // 2^30 max grid in x dimension
    int grid = (int)grid64;
    hardtanh_kernel<<<grid, block, 0, stream>>>(input, output, numel);
}
