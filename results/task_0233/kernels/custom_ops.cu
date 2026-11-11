#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel for argmin over dimension 1 (size 1)
// Each thread computes the index of the minimum value along the channel dimension
// for a given (n, h, w) location. Since dim=1 has size 1, the result is always 0.
// This kernel handles non-contiguous input via strides.
__global__ void argmin_dim1_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    int N,       // batch
    int C,       // channels (size of dim=1)
    int H,       // height
    int W,       // width
    int64_t stride_n,
    int64_t stride_c,
    int64_t stride_h,
    int64_t stride_w) {
    int64_t total = (int64_t)N * H * W;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int n = idx / (H * W);
        int rem = idx % (H * W);
        int h = rem / W;
        int w = rem % W;

        int64_t base = (int64_t)n * stride_n + h * stride_h + w * stride_w;
        int64_t min_idx = 0;
        float min_val = input[base + 0 * stride_c];
        // Loop over channels (currently C=1, but keep generic to be safe)
        for (int c = 1; c < C; ++c) {
            float val = input[base + c * stride_c];
            // torch.argmin returns first occurrence of minimum; use strict < to keep first
            if (val < min_val) {
                min_val = val;
                min_idx = c;
            }
        }
        output[idx] = min_idx;
    }
}

// CUDA kernel for bilinear 2D upsample with scale_factor and align_corners=false
// Each thread computes one output pixel.
// Input dimensions: N, C, H_in, W_in; output: N, C, H_out, W_out
// scale_factor is a float; output sizes are computed by caller as floor(H_in*scale) and floor(W_in*scale).
// align_corners false: src = (dst + 0.5) / scale - 0.5; clamp to [0, H_in-1]/[0, W_in-1].
__global__ void upsample_bilinear2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C,
    int H_in, int W_in,
    int H_out, int W_out,
    float scale_h, float scale_w,
    bool align_corners,
    int64_t stride_n,
    int64_t stride_c,
    int64_t stride_h,
    int64_t stride_w,
    int64_t out_stride_n,
    int64_t out_stride_c,
    int64_t out_stride_h,
    int64_t out_stride_w) {
    int64_t total = (int64_t)N * C * H_out * W_out;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int w_out = idx % W_out;
        int64_t tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        float src_h, src_w;
        if (align_corners) {
            // This path is not used by reference but kept for completeness if needed.
            float scale_h_c = (H_in > 1) ? (float)(H_in - 1) / (float)(H_out - 1) : 1.0f;
            float scale_w_c = (W_in > 1) ? (float)(W_in - 1) / (float)(W_out - 1) : 1.0f;
            src_h = h_out * scale_h_c;
            src_w = w_out * scale_w_c;
        } else {
            src_h = (h_out + 0.5f) / scale_h - 0.5f;
            src_w = (w_out + 0.5f) / scale_w - 0.5f;
            // clamp to [0, H_in-1] and [0, W_in-1]
            if (src_h < 0.0f) src_h = 0.0f;
            if (src_h > (float)(H_in - 1)) src_h = (float)(H_in - 1);
            if (src_w < 0.0f) src_w = 0.0f;
            if (src_w > (float)(W_in - 1)) src_w = (float)(W_in - 1);
        }

        int h0 = (int)src_h;
        int w0 = (int)src_w;
        int h1 = (h0 < H_in - 1) ? h0 + 1 : h0;
        int w1 = (w0 < W_in - 1) ? w0 + 1 : w0;
        float dh = src_h - h0;
        float dw = src_w - w0;

        int64_t in_base = (int64_t)n * stride_n + (int64_t)c * stride_c;
        float v00 = input[in_base + (int64_t)h0 * stride_h + (int64_t)w0 * stride_w];
        float v01 = input[in_base + (int64_t)h0 * stride_h + (int64_t)w1 * stride_w];
        float v10 = input[in_base + (int64_t)h1 * stride_h + (int64_t)w0 * stride_w];
        float v11 = input[in_base + (int64_t)h1 * stride_h + (int64_t)w1 * stride_w];

        float val = (1.0f - dh) * (1.0f - dw) * v00
                  + (1.0f - dh) * dw * v01
                  + dh * (1.0f - dw) * v10
                  + dh * dw * v11;

        int64_t out_base = (int64_t)n * out_stride_n + (int64_t)c * out_stride_c;
        output[out_base + (int64_t)h_out * out_stride_h + (int64_t)w_out * out_stride_w] = val;
    }
}

// Launcher for argmin along dim=1 (C dimension)
// Input: float pointer, output: int64_t pointer, dimensions and strides, stream
extern "C" void custom_argmin_launcher(
    const float* input,
    int64_t* output,
    int N, int C, int H, int W,
    int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
    cudaStream_t stream) {
    int64_t total = (int64_t)N * H * W;
    const int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL; // max grid dimension
    argmin_dim1_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input, output, N, C, H, W,
        stride_n, stride_c, stride_h, stride_w);
}

// Launcher for bilinear upsample 2D
// Input: float pointer, output: float pointer, dimensions and strides, scale factors, align_corners, stream
extern "C" void custom_upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    float scale_h, float scale_w,
    bool align_corners,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int64_t out_stride_n, int64_t out_stride_c, int64_t out_stride_h, int64_t out_stride_w,
    cudaStream_t stream) {
    int64_t total = (int64_t)N * C * H_out * W_out;
    const int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL;
    upsample_bilinear2d_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input, output,
        N, C, H_in, W_in, H_out, W_out,
        scale_h, scale_w, align_corners,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}
