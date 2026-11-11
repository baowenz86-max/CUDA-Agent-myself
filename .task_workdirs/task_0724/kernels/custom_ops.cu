#include <cuda_runtime.h>
#include <cmath>

// -----------------------------------------------------------------------------
// custom_mean_dim1: reduce over dim=1 (channels) with keepdim=True
// Input:  x shape [B, C, H, W] contiguous float32
// Output: y shape [B, 1, H, W] contiguous float32
// -----------------------------------------------------------------------------

__global__ void mean_dim1_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int B, int C, int H, int W) {
    int total = B * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int hw = H * W;
    int b = idx / hw;
    int hw_idx = idx - b * hw;

    const float* base = input + (size_t)b * C * hw + hw_idx;
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        sum += base[(size_t)c * hw];
    }
    output[idx] = sum / (float)C;
}

extern "C" void custom_mean_dim1_launcher(const float* input,
                                           float* output,
                                           int B, int C, int H, int W,
                                           cudaStream_t stream) {
    int total = B * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mean_dim1_kernel<<<blocks, threads, 0, stream>>>(input, output, B, C, H, W);
}

// -----------------------------------------------------------------------------
// custom_gelu: elementwise GELU (tanh approximation)
// y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Input:  x shape [N] contiguous float32 (flattened)
// Output: y shape [N] contiguous float32
// -----------------------------------------------------------------------------

__global__ void gelu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = input[idx];
    const float sqrt_2_over_pi = 0.7978845608028654f; // sqrt(2/pi)
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x3);
    float tanh_val = tanhf(inner);
    output[idx] = 0.5f * x * (1.0f + tanh_val);
}

extern "C" void custom_gelu_launcher(const float* input,
                                      float* output,
                                      int N,
                                      cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    gelu_kernel<<<blocks, threads, 0, stream>>>(input, output, N);
}

// No additional kernels or launchers needed.
