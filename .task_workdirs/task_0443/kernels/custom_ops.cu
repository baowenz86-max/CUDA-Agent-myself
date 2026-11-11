#include <cuda_runtime.h>
#include <cmath>

// GELU tanh approximation kernel
__global__ void gelu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    const float c1 = 0.7978845608028654f;  // sqrt(2/pi)
    const float c2 = 0.044715f;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = c1 * (x + c2 * x3);
        float tanh_val = tanhf(inner);
        output[i] = 0.5f * x * (1.0f + tanh_val);
    }
}

// Cumulative sum along dim=1
// Assumes x is 4D: [B, C, H, W]
// Each thread handles one (b, h, w) position and accumulates over C
__global__ void cumsum_dim1_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int64_t B, int64_t C, int64_t H, int64_t W) {
    int64_t total_threads = B * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t pos = idx; pos < total_threads; pos += stride) {
        int64_t w = pos % W;
        int64_t tmp = pos / W;
        int64_t h = tmp % H;
        int64_t b = tmp / H;

        float accum = 0.0f;
        int64_t base = (b * C + 0) * H * W + h * W + w;
        for (int64_t c = 0; c < C; ++c) {
            accum += input[base + c * H * W];
            output[base + c * H * W] = accum;
        }
    }
}

// Launchers (no PyTorch dependencies)
extern "C" void gelu_forward_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      cudaStream_t stream) {
    const int threads = 256;
    // Grid-stride loop, limit to reasonable grid size for large numel
    int64_t blocks_64 = (numel + threads - 1) / threads;
    int blocks = static_cast<int>(min(blocks_64, static_cast<int64_t>(65535)));
    gelu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void cumsum_dim1_forward_launcher(const float* input,
                                              float* output,
                                              int64_t B, int64_t C, int64_t H, int64_t W,
                                              cudaStream_t stream) {
    int64_t total_positions = B * H * W;
    const int threads = 256;
    int64_t blocks_64 = (total_positions + threads - 1) / threads;
    int blocks = static_cast<int>(min(blocks_64, static_cast<int64_t>(65535)));
    cumsum_dim1_kernel<<<blocks, threads, 0, stream>>>(input, output, B, C, H, W);
}
