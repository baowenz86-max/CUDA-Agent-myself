#include <cuda_runtime.h>

// Elementwise clamp kernel: output[i] = min(max(input[i], min_val), max_val)
__global__ void clamp_scalar_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel,
    float min_val,
    float max_val) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        val = fminf(fmaxf(val, min_val), max_val);
        output[i] = val;
    }
}

// Elementwise subtract scalar kernel: output[i] = input[i] - scalar
__global__ void sub_scalar_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel,
    float scalar) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = input[i] - scalar;
    }
}

// Helper to compute grid size with cap
static inline int compute_grid_size(int64_t numel, int block_size) {
    int64_t blocks = (numel + block_size - 1) / block_size;
    // Limit grid to a reasonable maximum for good occupancy and avoid excessive launch overhead
    const int max_blocks = 65535; // common launch bound
    if (blocks > max_blocks) blocks = max_blocks;
    return (int)blocks;
}

// Launcher for clamp_scalar
// Extern "C" so binding.cpp can declare and call it without name mangling
extern "C" void clamp_scalar_launcher(
    float* output,
    const float* input,
    int64_t numel,
    float min_val,
    float max_val,
    cudaStream_t stream) {
    const int block_size = 256;
    int grid_size = compute_grid_size(numel, block_size);
    clamp_scalar_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, numel, min_val, max_val);
}

// Launcher for sub_scalar
extern "C" void sub_scalar_launcher(
    float* output,
    const float* input,
    int64_t numel,
    float scalar,
    cudaStream_t stream) {
    const int block_size = 256;
    int grid_size = compute_grid_size(numel, block_size);
    sub_scalar_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, numel, scalar);
}
