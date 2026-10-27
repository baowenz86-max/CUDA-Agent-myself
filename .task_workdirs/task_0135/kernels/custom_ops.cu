// kernels/custom_ops.cu
// CUDA implementations for trunc and 2x2 stride-2 max pool.
// No PyTorch or C++ binding dependencies; only raw pointers, scalars, and cudaStream_t.

#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// trunc kernel: y[i] = truncf(x[i])
// Supports arbitrary strides via flattened linear index over numel.
// -----------------------------------------------------------------------------
__global__ void trunc_kernel(float* __restrict__ y,
                             const float* __restrict__ x,
                             int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; idx < numel; idx += stride) {
        y[idx] = truncf(x[idx]);
    }
}

// -----------------------------------------------------------------------------
// max_pool2d kernel for kernel_size=2, stride=2, padding=0, dilation=1.
// Input: [N, C, H, W], output: [N, C, Ho, Wo] where Ho = H/2, Wo = W/2 (H,W even).
// Assumes contiguous NCHW layout. The binding layer will make the input contiguous
// before calling this launcher if necessary.
// -----------------------------------------------------------------------------
__global__ void max_pool2d_2x2_stride2_kernel(const float* __restrict__ input,
                                              float* __restrict__ output,
                                              int64_t N,
                                              int64_t C,
                                              int64_t H,
                                              int64_t W,
                                              int64_t Ho,
                                              int64_t Wo) {
    // Each thread computes one output element.
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = N * C * Ho * Wo;
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (; idx < total; idx += stride) {
        // decompose idx into n, c, ho, wo
        int64_t tmp = idx;
        int64_t wo = tmp % Wo;
        tmp /= Wo;
        int64_t ho = tmp % Ho;
        tmp /= Ho;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // input coordinates for 2x2 window with stride 2
        int64_t h_base = ho * 2;
        int64_t w_base = wo * 2;

        // load four values
        float v0 = input[((n * C + c) * H + h_base) * W + w_base];
        float v1 = input[((n * C + c) * H + h_base) * W + w_base + 1];
        float v2 = input[((n * C + c) * H + (h_base + 1)) * W + w_base];
        float v3 = input[((n * C + c) * H + (h_base + 1)) * W + w_base + 1];

        // maximum
        float m = v0;
        m = (v1 > m) ? v1 : m;
        m = (v2 > m) ? v2 : m;
        m = (v3 > m) ? v3 : m;

        output[idx] = m;
    }
}

// -----------------------------------------------------------------------------
// Host launchers (C linkage)
// -----------------------------------------------------------------------------
extern "C" void trunc_launcher(float* output,
                               const float* input,
                               int64_t numel,
                               cudaStream_t stream) {
    const int block_size = 256;
    // Cap grid to avoid excessive blocks, use grid-stride loop.
    int64_t blocks_64 = (numel + block_size - 1) / block_size;
    const int64_t max_blocks = 65535;
    if (blocks_64 > max_blocks) blocks_64 = max_blocks;
    if (blocks_64 < 1) blocks_64 = 1;
    int blocks = static_cast<int>(blocks_64);
    trunc_kernel<<<blocks, block_size, 0, stream>>>(output, input, numel);
}

extern "C" void max_pool2d_launcher(const float* input,
                                    float* output,
                                    int64_t N,
                                    int64_t C,
                                    int64_t H,
                                    int64_t W,
                                    cudaStream_t stream) {
    int64_t Ho = H / 2;
    int64_t Wo = W / 2;
    if (Ho <= 0 || Wo <= 0) return;

    int64_t total = N * C * Ho * Wo;
    const int block_size = 256;
    int64_t blocks_64 = (total + block_size - 1) / block_size;
    const int64_t max_blocks = 65535;
    if (blocks_64 > max_blocks) blocks_64 = max_blocks;
    if (blocks_64 < 1) blocks_64 = 1;
    int blocks = static_cast<int>(blocks_64);

    max_pool2d_2x2_stride2_kernel<<<blocks, block_size, 0, stream>>>(
        input, output, N, C, H, W, Ho, Wo);
}
