#include <cuda_runtime.h>
#include <cstdint>

// Bilinear upsampling kernel (align_corners=False)
// input: float pointer with shape [N, C, H, W], non-contiguous supported via strides
// output: float pointer with shape [N, C, outH, outW], non-contiguous supported via strides
// scale_factor: integer upsampling factor >= 1
template <typename T>
__global__ void upsample_bilinear2d_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int scale_factor,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long total = (long long)N * C * outH * outW;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    float inv_scale = 1.0f / (float)scale_factor;

    for (long long linear = idx; linear < total; linear += stride) {
        int ow = linear % outW;
        long long tmp = linear / outW;
        int oh = tmp % outH;
        tmp /= outH;
        int c = tmp % C;
        int n = tmp / C;

        // source coordinate (align_corners=False)
        float src_h_f = (oh + 0.5f) * inv_scale - 0.5f;
        float src_w_f = (ow + 0.5f) * inv_scale - 0.5f;

        // clamp to valid range [0, H-1] and [0, W-1]
        src_h_f = fmaxf(0.0f, fminf(src_h_f, (float)(H - 1)));
        src_w_f = fmaxf(0.0f, fminf(src_w_f, (float)(W - 1)));

        int h0 = (int)src_h_f;
        int w0 = (int)src_w_f;
        int h1 = min(h0 + 1, H - 1);
        int w1 = min(w0 + 1, W - 1);

        float h_lambda = src_h_f - h0;
        float w_lambda = src_w_f - w0;

        const T* in_base = input + (long long)n * in_stride_n + (long long)c * in_stride_c;
        const T* p00 = in_base + (long long)h0 * in_stride_h + (long long)w0 * in_stride_w;
        const T* p01 = in_base + (long long)h0 * in_stride_h + (long long)w1 * in_stride_w;
        const T* p10 = in_base + (long long)h1 * in_stride_h + (long long)w0 * in_stride_w;
        const T* p11 = in_base + (long long)h1 * in_stride_h + (long long)w1 * in_stride_w;

        float val = (1.0f - h_lambda) * ((1.0f - w_lambda) * (*p00) + w_lambda * (*p01)) +
                    h_lambda * ((1.0f - w_lambda) * (*p10) + w_lambda * (*p11));

        T* out_ptr = output + (long long)n * out_stride_n + (long long)c * out_stride_c +
                     (long long)oh * out_stride_h + (long long)ow * out_stride_w;
        *out_ptr = (T)val;
    }
}

extern "C" void upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    int scale_factor,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * outH * outW;
    if (total == 0) return;

    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    // Cap blocks to avoid excessive grid dimensions; grid-stride loop handles the rest.
    const int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;

    upsample_bilinear2d_kernel<float><<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, H, W,
        outH, outW,
        scale_factor,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}
