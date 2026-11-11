#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel for nearest neighbor upsampling of 4D tensor (N, C, H, W)
// Supports arbitrary strides via input strides.
// Output is assumed contiguous with strides computed by binding layer.
// Index mapping: src_h = min((int)floor(dst_h / scale_h), H-1)
//               src_w = min((int)floor(dst_w / scale_w), W-1)
__global__ void upsample_nearest_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    float scale_h, float scale_w,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int64_t out_stride_n, int64_t out_stride_c, int64_t out_stride_h, int64_t out_stride_w)
{
    int64_t total = (int64_t)N * C * outH * outW;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t linear = idx; linear < total; linear += stride) {
        // Decode output indices
        int ow = linear % outW;
        int oh = (linear / outW) % outH;
        int oc = (linear / ((int64_t)outW * outH)) % C;
        int on = linear / ((int64_t)outW * outH * C);

        // Source indices
        int sh = (int)floorf((float)oh / scale_h);
        int sw = (int)floorf((float)ow / scale_w);
        sh = min(sh, H - 1);
        sw = min(sw, W - 1);

        // Input offset
        int64_t in_idx = (int64_t)on * in_stride_n +
                         (int64_t)oc * in_stride_c +
                         (int64_t)sh * in_stride_h +
                         (int64_t)sw * in_stride_w;

        // Output offset
        int64_t out_idx = (int64_t)on * out_stride_n +
                          (int64_t)oc * out_stride_c +
                          (int64_t)oh * out_stride_h +
                          (int64_t)ow * out_stride_w;

        output[out_idx] = input[in_idx];
    }
}

// Launcher (C interface). Called by binding.cpp with proper parameters.
extern "C" void custom_upsample_nearest_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    double scale_h, double scale_w,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int64_t out_stride_n, int64_t out_stride_c, int64_t out_stride_h, int64_t out_stride_w,
    cudaStream_t stream)
{
    float sh = (float)scale_h;
    float sw = (float)scale_w;

    int64_t total = (int64_t)N * C * outH * outW;
    if (total <= 0) return;

    // Grid-stride loop configuration
    const int block_size = 256;
    int64_t blocks_needed = (total + block_size - 1) / block_size;
    // Cap grid dimension to avoid oversized launch (max grid dim in x is 2^31-1)
    int grid_size = (int)min((int64_t)blocks_needed, (int64_t)65535);
    // 65535 is a safe cap for all GPUs; kernel uses grid-stride loop.

    upsample_nearest_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output,
        N, C, H, W,
        outH, outW,
        sh, sw,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}
