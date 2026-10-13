#include <cuda_runtime.h>
#include <cstdint>

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
        int oh = i % out_W;
        int64_t tmp = i / out_W;
        int oc = tmp % out_H;
        tmp /= out_H;
        int on = tmp % C;
        int n = tmp / C;

        // Reflect indices for height and width
        int ih = oh - pad_top;
        if (ih < 0) ih = -ih - 1;
        else if (ih >= H) ih = 2 * H - ih - 1;

        int iw = i % out_W - pad_left;
        if (iw < 0) iw = -iw - 1;
        else if (iw >= W) iw = 2 * W - iw - 1;

        // Compute source linear index using input strides
        int64_t src_idx = (int64_t)n * in_stride_n + (int64_t)on * in_stride_c
                       + (int64_t)ih * in_stride_h + (int64_t)iw * in_stride_w;
        // Compute destination linear index using output strides
        int64_t dst_idx = (int64_t)n * out_stride_n + (int64_t)on * out_stride_c
                        + (int64_t)oh * out_stride_h + (int64_t)oc * out_stride_w;
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

// Host launcher for reflection_pad2d
// x, out: raw device pointers (float32)
// N, C, H, W: input dims
// pad_left, pad_right, pad_top, pad_bottom
// in_strides: array of 4 int64 strides
// out_strides: array of 4 int64 strides
// stream: CUDA stream
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
    // Cap grid at 65535*4? Better use a reasonable max and rely on grid-stride loop.
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
// x: input pointer
// out: output pointer pre-zeroed (float32)
// N, C, H, W: dims
// in_strides: array of 4 int64 strides
// stream
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
