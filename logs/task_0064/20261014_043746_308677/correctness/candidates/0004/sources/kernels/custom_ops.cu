#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// ReflectionPad2d kernel
// x: input pointer (float32)
// out: output pointer (float32)
// N, C, H, W: input dimensions
// pad_left, pad_right, pad_top, pad_bottom: padding sizes
// in_stride_n, in_stride_c, in_stride_h, in_stride_w: input strides
// out_stride_n, out_stride_c, out_stride_h, out_stride_w: output strides
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int64_t out_stride_n, int64_t out_stride_c, int64_t out_stride_h, int64_t out_stride_w
) {
    int64_t total = (int64_t)N * C * (H + pad_top + pad_bottom) * (W + pad_left + pad_right);
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    int out_H = H + pad_top + pad_bottom;
    int out_W = W + pad_left + pad_right;

    for (int64_t i = idx; i < total; i += stride) {
        // Decompose flat index in NCHW order: ow, oh, c, n
        int ow = i % out_W;
        int64_t tmp = i / out_W;
        int oh = tmp % out_H;
        tmp /= out_H;
        int c = tmp % C;
        int n = tmp / C;

        // Reflect indices for height and width
        int ih = oh - pad_top;
        if (ih < 0) ih = -ih - 1;
        else if (ih >= H) ih = 2 * H - ih - 1;

        int iw = ow - pad_left;
        if (iw < 0) iw = -iw - 1;
        else if (iw >= W) iw = 2 * W - iw - 1;

        // Compute source linear index using input strides
        int64_t src_idx = (int64_t)n * in_stride_n + (int64_t)c * in_stride_c
                       + (int64_t)ih * in_stride_h + (int64_t)iw * in_stride_w;
        // Compute destination linear index using output strides
        int64_t dst_idx = (int64_t)n * out_stride_n + (int64_t)c * out_stride_c
                        + (int64_t)oh * out_stride_h + (int64_t)ow * out_stride_w;
        out[dst_idx] = x[src_idx];
    }
}

// nansum_dim12 kernel
// x: input pointer (float32)
// out: output pointer (float32) to be added to (must be pre-zeroed)
// N, C, H, W: dimensions
// in_stride_n, in_stride_c, in_stride_h, in_stride_w: input strides
__global__ void nansum_dim12_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w
) {
    int64_t total = (int64_t)N * C * H * W;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        int w = i % W;
        int64_t tmp = i / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;

        int64_t src_idx = (int64_t)n * in_stride_n + (int64_t)c * in_stride_c
                        + (int64_t)h * in_stride_h + (int64_t)w * in_stride_w;
        float val = x[src_idx];
        if (!isnan(val)) {
            atomicAdd(&out[(int64_t)n * W + w], val);
        }
    }
}

// conv2d kernel: direct convolution with loops over kernel and input channels
// x: input [N, C_in, H_in, W_in]
// weight: [C_out, C_in, KH, KW]
// bias: [C_out] (optional, may be null)
// out: [N, C_out, H_out, W_out]
// stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w
// in_strides: input strides (4)
// weight_strides: weight strides (4)
// out_strides: output strides (4)
__global__ void conv2d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C_in, int H_in, int W_in,
    int C_out, int KH, int KW,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int64_t weight_stride_o, int64_t weight_stride_i, int64_t weight_stride_h, int64_t weight_stride_w,
    int64_t out_stride_n, int64_t out_stride_c, int64_t out_stride_h, int64_t out_stride_w
) {
    int64_t total = (int64_t)N * C_out * H_out * W_out;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        int w_out = i % W_out;
        int64_t tmp = i / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c_out = tmp % C_out;
        int n = tmp / C_out;

        float sum = 0.0f;
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kh = 0; kh < KH; ++kh) {
                int h_in = h_out * stride_h - padding_h + kh * dilation_h;
                if (h_in < 0 || h_in >= H_in) continue;
                for (int kw = 0; kw < KW; ++kw) {
                    int w_in = w_out * stride_w - padding_w + kw * dilation_w;
                    if (w_in < 0 || w_in >= W_in) continue;
                    int64_t x_idx = (int64_t)n * in_stride_n + (int64_t)c_in * in_stride_c
                                    + (int64_t)h_in * in_stride_h + (int64_t)w_in * in_stride_w;
                    int64_t w_idx = (int64_t)c_out * weight_stride_o + (int64_t)c_in * weight_stride_i
                                    + (int64_t)kh * weight_stride_h + (int64_t)kw * weight_stride_w;
                    sum += x[x_idx] * weight[w_idx];
                }
            }
        }
        if (bias != nullptr) {
            sum += bias[c_out];
        }
        int64_t out_idx = (int64_t)n * out_stride_n + (int64_t)c_out * out_stride_c
                        + (int64_t)h_out * out_stride_h + (int64_t)w_out * out_stride_w;
        out[out_idx] = sum;
    }
}

// Host launcher for reflection_pad2d
extern "C" void reflection_pad2d_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    const int64_t* in_strides,
    const int64_t* out_strides,
    cudaStream_t stream
) {
    int64_t total = (int64_t)N * C * (H + pad_top + pad_bottom) * (W + pad_left + pad_right);
    if (total == 0) return;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    int maxGrid = 65535 * 4;
    if (grid > maxGrid) grid = maxGrid;
    reflection_pad2d_kernel<<<grid, block, 0, stream>>>(
        x, out, N, C, H, W,
        pad_left, pad_right, pad_top, pad_bottom,
        in_strides[0], in_strides[1], in_strides[2], in_strides[3],
        out_strides[0], out_strides[1], out_strides[2], out_strides[3]
    );
}

// Host launcher for nansum_dim12
extern "C" void nansum_dim12_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    const int64_t* in_strides,
    cudaStream_t stream
) {
    int64_t total = (int64_t)N * C * H * W;
    if (total == 0) return;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    int maxGrid = 65535 * 4;
    if (grid > maxGrid) grid = maxGrid;
    nansum_dim12_kernel<<<grid, block, 0, stream>>>(
        x, out, N, C, H, W,
        in_strides[0], in_strides[1], in_strides[2], in_strides[3]
    );
}

// Host launcher for conv2d
extern "C" void conv2d_launcher(
    const float* x,
    const float* weight,
    const float* bias,  // may be nullptr for no bias
    float* out,
    int N, int C_in, int H_in, int W_in,
    int C_out, int KH, int KW,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    const int64_t* in_strides,
    const int64_t* weight_strides,
    const int64_t* out_strides,
    cudaStream_t stream
) {
    int64_t total = (int64_t)N * C_out * H_out * W_out;
    if (total == 0) return;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    int maxGrid = 65535 * 4;
    if (grid > maxGrid) grid = maxGrid;
    conv2d_kernel<<<grid, block, 0, stream>>>(
        x, weight, bias, out,
        N, C_in, H_in, W_in,
        C_out, KH, KW,
        H_out, W_out,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        in_strides[0], in_strides[1], in_strides[2], in_strides[3],
        weight_strides[0], weight_strides[1], weight_strides[2], weight_strides[3],
        out_strides[0], out_strides[1], out_strides[2], out_strides[3]
    );
}
