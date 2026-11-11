#include <cuda_runtime.h>

// Softplus kernel: y = log(1 + exp(x)) with threshold for numerical stability
// The threshold is set to 20, matching PyTorch's default behavior.
__global__ void softplus_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        // Stable softplus: if x > threshold, return x (since exp(x) would overflow)
        // Else compute log1p(exp(x))
        output[i] = (x > 20.0f) ? x : log1pf(expf(x));
    }
}

// Launcher for softplus forward
extern "C" void softplus_launcher(const float* input,
                                  float* output,
                                  int64_t numel,
                                  cudaStream_t stream) {
    if (numel <= 0) return;
    const int block_size = 256;
    // Calculate grid size: enough blocks to cover numel, but cap at a reasonable number
    // to avoid launching too many blocks. Use grid-stride loop for large tensors.
    int64_t blocks = (numel + block_size - 1) / block_size;
    // Cap grid at 65535 (max grid dim in x for CC < 3.0, but for modern GPUs it's 2^31-1)
    // However, to avoid launching excessively large grids, we can cap at e.g. 4096 and rely on grid-stride loop.
    if (blocks > 4096) blocks = 4096;
    dim3 grid(blocks);
    dim3 block(block_size);
    softplus_kernel<<<grid, block, 0, stream>>>(input, output, numel);
}
