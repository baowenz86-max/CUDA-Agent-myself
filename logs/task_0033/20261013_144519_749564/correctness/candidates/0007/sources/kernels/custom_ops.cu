#include <cuda_runtime.h>

// Elementwise addcmul: output = input + value * tensor1 * tensor2
// All tensors are assumed to be float32 and contiguous or can be accessed linearly.
// Grid-stride loop ensures full coverage for large tensors.
__global__ void addcmul_kernel(
    const float* __restrict__ input,
    const float* __restrict__ tensor1,
    const float* __restrict__ tensor2,
    float* __restrict__ output,
    float value,
    int64_t numel)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; idx < numel; idx += stride) {
        output[idx] = input[idx] + value * tensor1[idx] * tensor2[idx];
    }
}

// Launcher for addcmul
// All pointers are raw device pointers, no PyTorch dependencies.
extern "C" void addcmul_forward(
    const float* input,
    const float* tensor1,
    const float* tensor2,
    float* output,
    float value,
    int64_t numel,
    cudaStream_t stream)
{
    if (numel <= 0) return;
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    // Cap grid size to avoid exceeding max grid dimension (2^31-1).
    // For very large sizes, grid-stride loop handles the rest.
    const int64_t max_grid_size = 2147483647LL;
    if (grid_size > max_grid_size) {
        grid_size = max_grid_size;
    }
    addcmul_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input, tensor1, tensor2, output, value, numel);
    // Error check can be done by binding layer after launch; we don't do it here to avoid cudaGetLastError overhead.
}
