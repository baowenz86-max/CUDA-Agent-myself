#include <cuda_runtime.h>
#include <cmath>

// GELU approximate tanh formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
__global__ void gelu_forward_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/pi)
    const float coeff = 0.044715f;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        float x_cube = x * x * x;
        float inner = sqrt_2_over_pi * (x + coeff * x_cube);
        float tanh_val = tanhf(inner);
        output[i] = 0.5f * x * (1.0f + tanh_val);
    }
}

extern "C" void gelu_forward_launcher(float* output, const float* input, int64_t numel, cudaStream_t stream) {
    if (numel <= 0) return;
    const int block_size = 256;
    // Cap grid to avoid exceeding max grid dimension (2^31-1) for large tensors; grid-stride loop handles rest.
    int64_t max_blocks_per_dim = 2147483647LL; // practical max for x dimension
    int64_t blocks = (numel + block_size - 1) / block_size;
    if (blocks > max_blocks_per_dim) blocks = max_blocks_per_dim;
    int grid = static_cast<int>(blocks);
    gelu_forward_kernel<<<grid, block_size, 0, stream>>>(input, output, numel);
}
