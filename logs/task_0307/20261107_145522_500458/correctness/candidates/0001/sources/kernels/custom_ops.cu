#include <cuda_runtime.h>
#include <cmath>

// GELU activation: x * 0.5 * (1 + erf(x / sqrt(2)))
__global__ void gelu_kernel(float* __restrict__ out, const float* __restrict__ inp, int64_t n) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float x = inp[i];
        out[i] = x * 0.5f * (1.0f + erff(x * 0.70710678118654752440f)); // 1/sqrt(2)
    }
}

// PixelShuffle for r=2: output shape [N, C_out, H*2, W*2]
// Input shape [N, C_in, H, W], where C_in = C_out * 4
// Mapping: out[n, c, h, w] = inp[n, c*4 + (h%2)*2 + (w%2), h/2, w/2]
__global__ void pixel_shuffle_kernel(float* __restrict__ out, const float* __restrict__ inp,
                                     int N, int C_out, int H, int W) {
    int64_t total = (int64_t)N * C_out * H * W * 4; // output elements
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        int w = i % (W * 2);
        int64_t tmp = i / (W * 2);
        int h = tmp % (H * 2);
        tmp /= (H * 2);
        int c = tmp % C_out;
        int n = tmp / C_out;

        int in_w = w / 2;
        int in_h = h / 2;
        int offset = (h % 2) * 2 + (w % 2);
        int in_c = c * 4 + offset;

        int64_t in_idx = (((int64_t)n * (C_out * 4) + in_c) * H + in_h) * W + in_w;
        out[i] = inp[in_idx];
    }
}

// Bilinear 2x upsample, align_corners=False
// Output size = (H*2, W*2)
// PyTorch coordinate mapping:
//   src_x = (dst_x + 0.5) * (in_w / out_w) - 0.5, clamped
// For scale=2: src_x = (dst_x + 0.5) * 0.5 - 0.5 = dst_x * 0.5 - 0.25
__global__ void upsample_bilinear2x_kernel(float* __restrict__ out, const float* __restrict__ inp,
                                           int N, int C, int H_in, int W_in) {
    int H_out = H_in * 2;
    int W_out = W_in * 2;
    int64_t total = (int64_t)N * C * H_out * W_out;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    float scale_h = (float)H_in / (float)H_out;
    float scale_w = (float)W_in / (float)W_out;

    for (int64_t i = idx; i < total; i += stride) {
        int ow = i % W_out;
        int64_t tmp = i / W_out;
        int oh = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        float src_x = (ow + 0.5f) * scale_w - 0.5f;
        float src_y = (oh + 0.5f) * scale_h - 0.5f;
        src_x = fmaxf(src_x, 0.0f);
        src_y = fmaxf(src_y, 0.0f);
        int x0 = (int)src_x;
        int y0 = (int)src_y;
        int x1 = (x0 + 1 < W_in) ? (x0 + 1) : x0;
        int y1 = (y0 + 1 < H_in) ? (y0 + 1) : y0;
        float dx = src_x - (float)x0;
        float dy = src_y - (float)y0;

        int64_t base = ((int64_t)n * C + c) * H_in * W_in;
        float v00 = inp[base + y0 * W_in + x0];
        float v01 = inp[base + y0 * W_in + x1];
        float v10 = inp[base + y1 * W_in + x0];
        float v11 = inp[base + y1 * W_in + x1];
        float top = v00 * (1.0f - dx) + v01 * dx;
        float bottom = v10 * (1.0f - dx) + v11 * dx;
        out[i] = top * (1.0f - dy) + bottom * dy;
    }
}

// Helper to compute grid size
static int get_grid_size(int64_t total, int block_size) {
    int64_t grid = (total + block_size - 1) / block_size;
    if (grid > 2147483647LL) return 2147483647;
    return (int)grid;
}

// Launchers

extern "C" void gelu_forward_launcher(float* output, const float* input, int64_t numel, cudaStream_t stream) {
    int block = 256;
    int grid = get_grid_size(numel, block);
    if (grid == 0) return;
    gelu_kernel<<<grid, block, 0, stream>>>(output, input, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Logging omitted; caller can check via cudaGetLastError, but we don't throw here.
    }
}

extern "C" void pixel_shuffle_forward_launcher(float* output, const float* input,
                                                int N, int C_out, int H, int W,
                                                cudaStream_t stream) {
    int64_t total = (int64_t)N * C_out * H * W * 4;
    int block = 256;
    int grid = get_grid_size(total, block);
    if (grid == 0) return;
    pixel_shuffle_kernel<<<grid, block, 0, stream>>>(output, input, N, C_out, H, W);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // do nothing, binding will check?
    }
}

extern "C" void upsample_bilinear2x_forward_launcher(float* output, const float* input,
                                                      int N, int C, int H_in, int W_in,
                                                      cudaStream_t stream) {
    int64_t total = (int64_t)N * C * (H_in * 2) * (W_in * 2);
    int block = 256;
    int grid = get_grid_size(total, block);
    if (grid == 0) return;
    upsample_bilinear2x_kernel<<<grid, block, 0, stream>>>(output, input, N, C, H_in, W_in);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // do nothing
    }
}
