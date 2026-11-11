#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel: bitwise xor of abs(x) and y after conversion to int64, result as float32
__global__ void bitwise_xor_fp32_kernel(const float* __restrict__ x,
                                       const float* __restrict__ y,
                                       float* __restrict__ out,
                                       int64_t numel) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; i < numel; i += stride) {
        float x_abs = fabsf(x[i]);
        int64_t xi = static_cast<int64_t>(x_abs); // trunc toward zero, matches .type(torch.int64)
        int64_t yi = static_cast<int64_t>(y[i]);
        int64_t res = xi ^ yi;
        out[i] = static_cast<float>(res);
    }
}

// Launcher function, to be called from binding.cpp
extern "C" void bitwise_xor_fp32_custom_launcher(const float* x,
                                                const float* y,
                                                float* out,
                                                int64_t numel,
                                                cudaStream_t stream) {
    const int block_size = 256;
    // Compute grid size, but cap at reasonable number for grid-stride loop
    int64_t blocks = (numel + block_size - 1) / block_size;
    // Cap grid to avoid exceeding max grid dimension (usually 2^31-1 for x)
    // Use a reasonable cap: 65535 is safe but may be low; use 1<<20 for large data
    int max_blocks = 1 << 20; // ~1M blocks, provides good occupancy
    if (blocks > max_blocks) blocks = max_blocks;
    if (blocks == 0) return; // no elements
    bitwise_xor_fp32_kernel<<<(int)blocks, block_size, 0, stream>>>(x, y, out, numel);
}
