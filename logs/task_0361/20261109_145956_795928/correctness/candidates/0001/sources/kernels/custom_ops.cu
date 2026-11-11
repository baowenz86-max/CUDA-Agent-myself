#include <cuda_runtime.h>

// Hardswish activation: y = x * relu6(x + 3) / 6
__global__ void hardswish_kernel(float* __restrict__ output, const float* __restrict__ input, int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        float relu6 = fminf(fmaxf(x + 3.0f, 0.0f), 6.0f);
        output[i] = x * relu6 / 6.0f;
    }
}

// Launcher for elementwise hardswish on float32 tensors.
// Handles non-contiguous tensors via linear indexing (caller must allocate output with same layout).
extern "C" void hardswish_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream
) {
    const int block_size = 256;
    int64_t blocks = (numel + block_size - 1) / block_size;
    // Cap grid to avoid exceeding limits; grid-stride loop covers the rest.
    const int64_t max_grid = 65535;
    if (blocks > max_grid) blocks = max_grid;
    hardswish_kernel<<<static_cast<unsigned int>(blocks), block_size, 0, stream>>>(output, input, numel);
}
