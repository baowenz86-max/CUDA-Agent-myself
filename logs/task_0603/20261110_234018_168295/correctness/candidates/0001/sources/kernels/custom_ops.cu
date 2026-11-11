#include <cuda_runtime.h>

// GELU activation: y = 0.5 * x * (1 + erf(x / sqrt(2)))
// Supports arbitrary strides.
__global__ void gelu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t total_elements,
                            int64_t d1, int64_t d2, int64_t d3,
                            int64_t s0, int64_t s1, int64_t s2, int64_t s3) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        // Compute multi-dimensional indices from linear index (assuming no padding, but strides may be arbitrary)
        int64_t n = i / (d1 * d2 * d3);
        int64_t rem = i % (d1 * d2 * d3);
        int64_t c = rem / (d2 * d3);
        rem = rem % (d2 * d3);
        int64_t h = rem / d3;
        int64_t w = rem % d3;

        float x = input[n * s0 + c * s1 + h * s2 + w * s3];
        // GELU approximation: tanh is faster than erf. To match PyTorch F.gelu, we use erf.
        // PyTorch default approximate='none' uses erf.
        float gelu = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
        output[n * s0 + c * s1 + h * s2 + w * s3] = gelu;
    }
}

// Mean along dim=1: output[n, h, w] = sum_{c=0}^{d1-1} input[n, c, h, w] / d1
// Supports arbitrary strides.
__global__ void mean_dim1_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int64_t batch, int64_t d1, int64_t d2, int64_t d3,
                                 int64_t s0, int64_t s1, int64_t s2, int64_t s3,
                                 int64_t out_s0, int64_t out_s1, int64_t out_s2) {
    // Each thread handles one output element (n, h, w)
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch * d2 * d3;
    if (idx >= total) return;

    int64_t n = idx / (d2 * d3);
    int64_t rem = idx % (d2 * d3);
    int64_t h = rem / d3;
    int64_t w = rem % d3;

    float sum = 0.0f;
    const float* base = input + n * s0 + h * s2 + w * s3;
    for (int64_t c = 0; c < d1; ++c) {
        sum += base[c * s1];
    }
    output[n * out_s0 + h * out_s1 + w * out_s2] = sum / (float)d1;
}

// Launcher for GELU forward. Assumes output tensor already allocated with same shape as input.
extern "C" void gelu_forward_launcher(const float* input,
                                      float* output,
                                      int64_t total_elements,
                                      int64_t d1, int64_t d2, int64_t d3,
                                      int64_t s0, int64_t s1, int64_t s2, int64_t s3,
                                      cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    // Grid-stride loop allows up to maximum grid dimension, but we cap to avoid excessive blocks.
    // For 268M elements, 1M blocks is okay (max grid dim x = 2^31-1). Use 1D grid.
    if (blocks > 0) {
        gelu_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements,
                                                   d1, d2, d3, s0, s1, s2, s3);
    }
}

// Launcher for mean_dim1 forward. Assumes output tensor already allocated with shape [batch, d2, d3].
extern "C" void mean_dim1_forward_launcher(const float* input,
                                           float* output,
                                           int64_t batch, int64_t d1, int64_t d2, int64_t d3,
                                           int64_t s0, int64_t s1, int64_t s2, int64_t s3,
                                           int64_t out_s0, int64_t out_s1, int64_t out_s2,
                                           cudaStream_t stream) {
    int64_t total = batch * d2 * d3;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 0) {
        mean_dim1_kernel<<<blocks, threads, 0, stream>>>(input, output,
                                                         batch, d1, d2, d3,
                                                         s0, s1, s2, s3,
                                                         out_s0, out_s1, out_s2);
    }
}
