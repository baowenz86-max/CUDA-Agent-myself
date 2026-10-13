#include <cuda_runtime.h>

// Sigmoid kernel: applies 1/(1+exp(-x)) elementwise.
// Grid-stride loop for arbitrary numel.
__global__ void sigmoid_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        // Stable sigmoid: 1 / (1 + exp(-val))
        output[i] = 1.0f / (1.0f + __expf(-val));
    }
}

// Launcher for sigmoid_forward. Assumes input is float32 contiguous CUDA tensor.
// output_ptr must already be allocated (by binding) with same shape.
extern "C" void sigmoid_forward_launcher(const float* input,
                                         float* output,
                                         int64_t numel,
                                         cudaStream_t stream) {
    if (numel <= 0) return;
    const int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    // Cap blocks to a reasonable maximum to avoid too many blocks (grid-stride will handle remainder).
    const int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    sigmoid_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}
