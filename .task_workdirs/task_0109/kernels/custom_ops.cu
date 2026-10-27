#include <cuda_runtime.h>
#include <cstdint>

// Helper for bilinear interpolation, align_corners=False
// Source coordinates: src = (dst + 0.5) / scale - 0.5
// Clamp to [0, dim-1], compute floor and ceil with weight.
__global__ void bilinear_upsample_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = N * C * outH * outW;

    for (int linear_idx = idx; linear_idx < total; linear_idx += stride) {
        // Decode linear index into n,c,oh,ow
        int ow = linear_idx % outW;
        int tmp = linear_idx / outW;
        int oh = tmp % outH;
        tmp /= outH;
        int c = tmp % C;
        int n = tmp / C;

        // Compute source coordinates
        float src_y = (static_cast<float>(oh) + 0.5f) / scale_factor - 0.5f;
        float src_x = (static_cast<float>(ow) + 0.5f) / scale_factor - 0.5f;

        // Clamp to [0, H-1] and [0, W-1]
        if (src_y < 0.f) src_y = 0.f;
        if (src_x < 0.f) src_x = 0.f;
        if (src_y > H - 1.f) src_y = H - 1.f;
        if (src_x > W - 1.f) src_x = W - 1.f;

        int y0 = static_cast<int>(src_y);
        int x0 = static_cast<int>(src_x);
        // Ensure within bounds after clamp (can be H-1 or W-1)
        int y1 = (y0 + 1 < H) ? (y0 + 1) : (H - 1);
        int x1 = (x0 + 1 < W) ? (x0 + 1) : (W - 1);

        float ly = src_y - y0;
        float lx = src_x - x0;

        // Input base for this n,c
        const float* in_nc = input + (static_cast<size_t>(n) * C + c) * H * W;

        float v00 = in_nc[y0 * W + x0];
        float v01 = in_nc[y0 * W + x1];
        float v10 = in_nc[y1 * W + x0];
        float v11 = in_nc[y1 * W + x1];

        // Bilinear interpolation
        float top = v00 * (1.f - lx) + v01 * lx;
        float bottom = v10 * (1.f - lx) + v11 * lx;
        float value = top * (1.f - ly) + bottom * ly;

        // Output index
        output[linear_idx] = value;
    }
}

__global__ void relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = val > 0.f ? val : 0.f;
    }
}

extern "C" void custom_upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    float scale_factor,
    cudaStream_t stream) {
    int outH = static_cast<int>(floorf(H * scale_factor));
    int outW = static_cast<int>(floorf(W * scale_factor));
    if (outH <= 0 || outW <= 0) {
        return; // No output elements; should not happen for valid inputs
    }
    int total = N * C * outH * outW;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Cap blocks to avoid launching too many (though for 201M elements, blocks ~786k, can be huge but allowed?)
    // Grid-stride loop handles all elements; use a reasonable grid size, e.g., min(blocks, 65535*?) Actually grid dimension max is 2^31-1 in x, so fine.
    // But to reduce launch overhead and allow dynamic parallelism? We'll just use the calculated blocks.
    // However, for very large outputs, blocks can be >2^31? total ~201M, /256 ~786k, well within limit.
    bilinear_upsample_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, outH, outW, scale_factor);
}

extern "C" void custom_relu_launcher(
    const float* input,
    float* output,
    size_t numel,
    cudaStream_t stream) {
    const int threads = 256;
    size_t blocks = (numel + threads - 1) / threads;
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}
