#include <cuda_runtime.h>

// Element-wise multiplication kernel: out[i] = x[i] * y[i]
__global__ void mul_tensor_kernel(float* __restrict__ out,
                                  const float* __restrict__ x,
                                  const float* __restrict__ y,
                                  int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        out[i] = x[i] * y[i];
    }
}

// Launcher: simple grid-stride loop over all elements.
// The binding layer ensures tensors are contiguous, same shape, float32, and on CUDA.
extern "C" void mul_tensor_launcher(float* out,
                                    const float* x,
                                    const float* y,
                                    int64_t n,
                                    cudaStream_t stream) {
    if (n == 0) return;
    const int block_size = 256;
    // Approximately enough blocks to fill the GPU; the grid-stride loop handles the rest.
    int64_t num_blocks = (n + block_size - 1) / block_size;
    // Cap blocks to a reasonable maximum to avoid launch failure (e.g., 2^31-1 blocks).
    const int64_t max_blocks = 65535 * 32;  // arbitrary large number, but safe
    if (num_blocks > max_blocks) num_blocks = max_blocks;
    mul_tensor_kernel<<<(int)num_blocks, block_size, 0, stream>>>(out, x, y, n);
}
