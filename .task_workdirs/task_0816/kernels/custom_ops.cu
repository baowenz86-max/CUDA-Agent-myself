#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// Bilinear upsampling kernel: output tensor layout NCHW, contiguous
// input: NCHW (may be non-contiguous, use strides)
// output: NCHW contiguous, computed as floor(in_h*scale), floor(in_w*scale)
// scale: float scale factor (typically 2.0)
__global__ void upsample_bilinear2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int in_h, int in_w,
    int out_h, int out_w,
    float scale_factor,
    int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
    int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * out_h * out_w;
    if (idx >= total) return;

    // decompose linear index into n,c,oh,ow
    int ow = idx % out_w;
    int tmp = idx / out_w;
    int oh = tmp % out_h;
    tmp /= out_h;
    int c = tmp % C;
    int n = tmp / C;

    // compute source coordinates (align_corners=False)
    float src_h = (oh + 0.5f) / scale_factor - 0.5f;
    float src_w = (ow + 0.5f) / scale_factor - 0.5f;

    int h0 = (int)floorf(src_h);
    int w0 = (int)floorf(src_w);
    int h1 = h0 + 1;
    int w1 = w0 + 1;

    float h1_lambda = src_h - h0;
    float w1_lambda = src_w - w0;
    float h0_lambda = 1.0f - h1_lambda;
    float w0_lambda = 1.0f - w1_lambda;

    // clamp to valid range for edges
    h0 = max(0, min(h0, in_h - 1));
    h1 = max(0, min(h1, in_h - 1));
    w0 = max(0, min(w0, in_w - 1));
    w1 = max(0, min(w1, in_w - 1));

    // gather four values
    const float* base = input + n * in_n_stride + c * in_c_stride;
    float v00 = base[h0 * in_h_stride + w0 * in_w_stride];
    float v01 = base[h0 * in_h_stride + w1 * in_w_stride];
    float v10 = base[h1 * in_h_stride + w0 * in_w_stride];
    float v11 = base[h1 * in_h_stride + w1 * in_w_stride];

    // bilinear interpolation
    float val = h0_lambda * (w0_lambda * v00 + w1_lambda * v01) +
                h1_lambda * (w0_lambda * v10 + w1_lambda * v11);

    output[n * out_n_stride + c * out_c_stride + oh * out_h_stride + ow * out_w_stride] = val;
}

// Mean reduction over channel dimension, keepdim=True
// input: NCHW (may be non-contiguous)
// output: N x 1 x H x W (contiguous)
__global__ void mean_dim1_keepdim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
    int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W;
    if (idx >= total) return;

    // decompose to n, h, w
    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    int n = tmp / H;

    float sum = 0.0f;
    const float* base = input + n * in_n_stride;
    for (int c = 0; c < C; ++c) {
        sum += base[c * in_c_stride + h * in_h_stride + w * in_w_stride];
    }
    float mean = sum / (float)C;

    output[n * out_n_stride + h * out_h_stride + w * out_w_stride] = mean;
}

// Host launcher for bilinear upsample
// input: device pointer to input tensor (float32), contiguous or strided
// output: device pointer to output tensor (float32), contiguous
// N,C,in_h,in_w: input dims
// out_h,out_w: output dims
// scale_factor: float scale
// in_strides: array of 4 ints (N,C,H,W strides in elements)
// out_strides: array of 4 ints (N,C,H,W strides in elements, should be contiguous)
extern "C" void upsample_bilinear2d_launcher(
    const float* input, float* output,
    int N, int C, int in_h, int in_w, int out_h, int out_w,
    float scale_factor,
    const int64_t* in_strides, const int64_t* out_strides,
    cudaStream_t stream)
{
    int total = N * C * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    upsample_bilinear2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, in_h, in_w, out_h, out_w, scale_factor,
        (int)in_strides[0], (int)in_strides[1], (int)in_strides[2], (int)in_strides[3],
        (int)out_strides[0], (int)out_strides[1], (int)out_strides[2], (int)out_strides[3]);
}

// Host launcher for channel mean (keepdim)
// input: device pointer to input tensor (float32), contiguous or strided
// output: device pointer to output tensor (float32), contiguous, shape [N,1,H,W]
// N,C,H,W: input dims
// in_strides, out_strides: as above
extern "C" void mean_dim1_keepdim_launcher(
    const float* input, float* output,
    int N, int C, int H, int W,
    const int64_t* in_strides, const int64_t* out_strides,
    cudaStream_t stream)
{
    int total = N * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    mean_dim1_keepdim_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W,
        (int)in_strides[0], (int)in_strides[1], (int)in_strides[2], (int)in_strides[3],
        (int)out_strides[0], (int)out_strides[1], (int)out_strides[2], (int)out_strides[3]);
}
