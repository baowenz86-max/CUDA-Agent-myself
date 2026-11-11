#include <cuda_runtime.h>
#include <cstdint>

// ReLU kernel: element-wise max(0, x)
__global__ void relu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = fmaxf(input[i], 0.0f);
    }
}

// Sum along dim=1 kernel for input shape [B, C, H, W], dim=1 has size C.
// Each thread computes one output element (b, h, w) by summing across C channels.
__global__ void sum_dim1_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int64_t B, int64_t C, int64_t H, int64_t W,
                                 int64_t total_output_elements) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < total_output_elements; i += stride) {
        // Decode output index i into (b, h, w)
        int64_t w = i % W;
        int64_t tmp = i / W;
        int64_t h = tmp % H;
        int64_t b = tmp / H;

        float sum = 0.0f;
        #pragma unroll
        for (int64_t c = 0; c < C; ++c) {
            int64_t in_idx = ((b * C + c) * H + h) * W + w;
            sum += input[in_idx];
        }
        output[i] = sum;
    }
}

// Relu launcher: C interface for binding.cpp
extern "C" void relu_forward_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      cudaStream_t stream) {
    if (numel <= 0) return;
    const int block_size = 256;
    // Use grid-stride loop, grid can be smaller than numel/block_size to limit grid size.
    int64_t max_blocks = (numel + block_size - 1) / block_size;
    // Cap grid at a reasonable maximum to avoid excessive blocks; grid-stride loop handles rest.
    int64_t grid_size = min(max_blocks, (int64_t)65535 * 32);
    relu_kernel<<<(int)grid_size, block_size, 0, stream>>>(input, output, numel);
}

// Sum dim=1 launcher: C interface for binding.cpp
// Input shape: [B, C, H, W], output shape: [B, H, W]
extern "C" void sum_dim1_launcher(const float* input,
                                  float* output,
                                  int64_t B, int64_t C, int64_t H, int64_t W,
                                  cudaStream_t stream) {
    int64_t total_output = B * H * W;
    if (total_output <= 0) return;
    const int block_size = 256;
    int64_t max_blocks = (total_output + block_size - 1) / block_size;
    int64_t grid_size = min(max_blocks, (int64_t)65535 * 32);
    sum_dim1_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        input, output, B, C, H, W, total_output);
}
