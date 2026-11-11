#include <cuda_runtime.h>

// Elementwise Leaky ReLU kernel with grid-stride loop.
// Each thread processes multiple elements if numel exceeds grid size.
__global__ void leaky_relu_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int64_t numel,
                                  float negative_slope) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = (val >= 0.0f) ? val : (negative_slope * val);
    }
}

// Host launcher: raw pointers, scalar negative_slope, total element count,
// and the current CUDA stream. No PyTorch dependencies.
extern "C" void leaky_relu_forward_launcher(const float* input,
                                            float* output,
                                            int64_t numel,
                                            float negative_slope,
                                            cudaStream_t stream) {
    if (numel <= 0) return;

    const int threads = 256;
    // Cap grid to avoid exceeding maximum grid dimension, use grid-stride loop.
    int64_t blocks64 = (numel + threads - 1) / threads;
    const int64_t max_blocks = 65535; // typical maximum for x dimension
    if (blocks64 > max_blocks) blocks64 = max_blocks;
    int grid = static_cast<int>(blocks64);

    leaky_relu_kernel<<<grid, threads, 0, stream>>>(input, output, numel, negative_slope);
}
