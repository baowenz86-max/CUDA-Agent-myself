#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel for single 3D average pooling.
// input: [N, C, D, H, W] contiguous float32
// output: [N, C, outD, outH, outW] contiguous float32
// kernel_d, kernel_h, kernel_w: pooling kernel sizes
// stride, pad_d, pad_h, pad_w: pooling params
// count_include_pad = true (denominator = kernel volume)
__global__ void avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int outD, int outH, int outW,
    int kernel_d, int kernel_h, int kernel_w,
    int stride, int pad_d, int pad_h, int pad_w)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * outD * outH * outW;
    if (tid >= total) return;

    // Decompose linear index into n, c, od, oh, ow
    int ow = tid % outW;
    int64_t tmp = tid / outW;
    int oh = tmp % outH;
    tmp /= outH;
    int od = tmp % outD;
    tmp /= outD;
    int c = tmp % C;
    int n = tmp / C;

    // Base pointer for this N,C slice
    const float* in_nc = input + ((int64_t)n * C + c) * (int64_t)D * H * W;

    float sum = 0.0f;
    int d_start = od * stride - pad_d;
    int h_start = oh * stride - pad_h;
    int w_start = ow * stride - pad_w;

    for (int kd = 0; kd < kernel_d; ++kd) {
        int d = d_start + kd;
        if (d < 0 || d >= D) continue;
        for (int kh = 0; kh < kernel_h; ++kh) {
            int h = h_start + kh;
            if (h < 0 || h >= H) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int w = w_start + kw;
                if (w < 0 || w >= W) continue;
                sum += in_nc[((int64_t)d * H + h) * W + w];
            }
        }
    }

    int denominator = kernel_d * kernel_h * kernel_w;
    output[tid] = sum / denominator;
}

// CUDA kernel for fused double 3D average pooling.
// Computes pool1 then pool2, writing final result directly.
// Input dimensions: [N, C, D, H, W]
// kernel1: (k1_d, k1_h, k1_w), kernel2: (k2_d, k2_h, k2_w)
// same stride and padding for both stages.
__global__ void avg_pool3d_double_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int out1D, int out1H, int out1W,
    int out2D, int out2H, int out2W,
    int k1_d, int k1_h, int k1_w,
    int k2_d, int k2_h, int k2_w,
    int stride, int pad_d, int pad_h, int pad_w)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * out2D * out2H * out2W;
    if (tid >= total) return;

    // Decompose into output indices (final pool output)
    int ow2 = tid % out2W;
    int64_t tmp = tid / out2W;
    int oh2 = tmp % out2H;
    tmp /= out2H;
    int od2 = tmp % out2D;
    tmp /= out2D;
    int c = tmp % C;
    int n = tmp / C;

    // Input base pointer for this N,C
    const float* in_nc = input + ((int64_t)n * C + c) * (int64_t)D * H * W;

    // For each position in final output, compute average over second pooling window
    float sum = 0.0f;
    int denom2 = k2_d * k2_h * k2_w;

    // Second pooling window start positions in intermediate (first pooling output) coordinates
    int d2_start = od2 * stride - pad_d;
    int h2_start = oh2 * stride - pad_h;
    int w2_start = ow2 * stride - pad_w;

    for (int k2d = 0; k2d < k2_d; ++k2d) {
        int d1 = d2_start + k2d;
        if (d1 < 0 || d1 >= out1D) continue;
        for (int k2h = 0; k2h < k2_h; ++k2h) {
            int h1 = h2_start + k2h;
            if (h1 < 0 || h1 >= out1H) continue;
            for (int k2w = 0; k2w < k2_w; ++k2w) {
                int w1 = w2_start + k2w;
                if (w1 < 0 || w1 >= out1W) continue;

                // Compute value of intermediate tensor at (d1, h1, w1) by first pooling
                float pool1_sum = 0.0f;
                int d_start = d1 * stride - pad_d;
                int h_start = h1 * stride - pad_h;
                int w_start = w1 * stride - pad_w;

                for (int k1d = 0; k1d < k1_d; ++k1d) {
                    int d = d_start + k1d;
                    if (d < 0 || d >= D) continue;
                    for (int k1h = 0; k1h < k1_h; ++k1h) {
                        int h = h_start + k1h;
                        if (h < 0 || h >= H) continue;
                        for (int k1w = 0; k1w < k1_w; ++k1w) {
                            int w = w_start + k1w;
                            if (w < 0 || w >= W) continue;
                            pool1_sum += in_nc[((int64_t)d * H + h) * W + w];
                        }
                    }
                }
                int denom1 = k1_d * k1_h * k1_w;
                sum += pool1_sum / denom1;
            }
        }
    }

    output[tid] = sum / denom2;
}

// Host launcher for single avg_pool3d
// Parameters:
//   input: float* device pointer to contiguous [N,C,D,H,W]
//   output: float* device pointer to preallocated [N,C,outD,outH,outW]
//   N, C, D, H, W: input dims
//   kernel_d, kernel_h, kernel_w: kernel sizes
//   stride, pad_d, pad_h, pad_w: pooling params
//   stream: CUDA stream
extern "C" void avg_pool3d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    int kernel_d, int kernel_h, int kernel_w,
    int stride, int pad_d, int pad_h, int pad_w,
    cudaStream_t stream)
{
    int outD = (D + 2 * pad_d - kernel_d) / stride + 1;
    int outH = (H + 2 * pad_h - kernel_h) / stride + 1;
    int outW = (W + 2 * pad_w - kernel_w) / stride + 1;

    int64_t total = (int64_t)N * C * outD * outH * outW;
    if (total <= 0) return;

    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    avg_pool3d_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, D, H, W,
        outD, outH, outW,
        kernel_d, kernel_h, kernel_w,
        stride, pad_d, pad_h, pad_w);
}

// Host launcher for fused double avg_pool3d
// Parameters:
//   input: float* device pointer to contiguous [N,C,D,H,W]
//   output: float* device pointer to preallocated [N,C,out2D,out2H,out2W]
//   N, C, D, H, W: input dims
//   k1_d, k1_h, k1_w, k2_d, k2_h, k2_w: kernel sizes for stage 1 and 2
//   stride, pad_d, pad_h, pad_w: pooling params (same for both stages)
//   stream: CUDA stream
extern "C" void avg_pool3d_double_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    int k1_d, int k1_h, int k1_w,
    int k2_d, int k2_h, int k2_w,
    int stride, int pad_d, int pad_h, int pad_w,
    cudaStream_t stream)
{
    int out1D = (D + 2 * pad_d - k1_d) / stride + 1;
    int out1H = (H + 2 * pad_h - k1_h) / stride + 1;
    int out1W = (W + 2 * pad_w - k1_w) / stride + 1;

    int out2D = (out1D + 2 * pad_d - k2_d) / stride + 1;
    int out2H = (out1H + 2 * pad_h - k2_h) / stride + 1;
    int out2W = (out1W + 2 * pad_w - k2_w) / stride + 1;

    int64_t total = (int64_t)N * C * out2D * out2H * out2W;
    if (total <= 0) return;

    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    avg_pool3d_double_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, D, H, W,
        out1D, out1H, out1W,
        out2D, out2H, out2W,
        k1_d, k1_h, k1_w,
        k2_d, k2_h, k2_w,
        stride, pad_d, pad_h, pad_w);
}
