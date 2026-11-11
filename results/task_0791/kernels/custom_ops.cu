#include <cuda_runtime.h>

// Fused kernel: out[i] = (float)(x1[i] || x2[i]);
// Since inputs are bool and output is float32, logical_or yields 0/1.
__global__ void logical_or_round_kernel(
    const bool* __restrict__ x1,
    const bool* __restrict__ x2,
    float* __restrict__ out,
    int64_t numel)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        out[i] = (x1[i] || x2[i]) ? 1.0f : 0.0f;
    }
}

extern "C" void logical_or_round_launcher(
    const bool* x1,
    const bool* x2,
    float* out,
    int64_t numel,
    cudaStream_t stream)
{
    const int threads = 256;
    // Cap grid at 65535*threads coverage via grid-stride loop; for large numel, grid is limited.
    int64_t max_blocks = 65535;
    int64_t needed_blocks = (numel + threads - 1) / threads;
    int blocks = static_cast<int>(needed_blocks < max_blocks ? needed_blocks : max_blocks);
    if (blocks < 1) blocks = 1;
    logical_or_round_kernel<<<blocks, threads, 0, stream>>>(x1, x2, out, numel);
}
