#include <cuda_runtime.h>

__global__ void upsample_bilinear2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    float rheight, float rwidth)
{
    int total = N * C * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int linear_idx = idx; linear_idx < total; linear_idx += stride) {
        int tmp = linear_idx;
        int x_out = tmp % outW;
        tmp /= outW;
        int y_out = tmp % outH;
        tmp /= outH;
        int c = tmp % C;
        tmp /= C;
        int n = tmp;

        float src_y = (outH > 1) ? (y_out + 0.5f) * rheight - 0.5f : 0.0f;
        float src_x = (outW > 1) ? (x_out + 0.5f) * rwidth - 0.5f : 0.0f;

        if (src_y < 0.0f) src_y = 0.0f;
        if (src_x < 0.0f) src_x = 0.0f;
        if (src_y > H - 1.0f) src_y = H - 1.0f;
        if (src_x > W - 1.0f) src_x = W - 1.0f;

        int y0 = (int)src_y;
        int x0 = (int)src_x;
        int y1 = (y0 < H - 1) ? y0 + 1 : y0;
        int x1 = (x0 < W - 1) ? x0 + 1 : x0;

        float ly = src_y - y0;
        float lx = src_x - x0;
        float hy = 1.0f - ly;
        float hx = 1.0f - lx;

        const float* in_nc = input + ((long long)n * C + c) * H * W;
        float v00 = in_nc[y0 * W + x0];
        float v01 = in_nc[y0 * W + x1];
        float v10 = in_nc[y1 * W + x0];
        float v11 = in_nc[y1 * W + x1];

        float val = hy * hx * v00 + hy * lx * v01 + ly * hx * v10 + ly * lx * v11;
        output[linear_idx] = val;
    }
}

__global__ void logical_and_range_kernel(
    const float* __restrict__ input,
    bool* __restrict__ output,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        float val = input[i];
        output[i] = (val > 0.0f) && (val < 1.0f);
    }
}

extern "C" void upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    float scale_factor,
    cudaStream_t stream)
{
    int total = N * C * outH * outW;
    if (total == 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    // align_corners=False: scale should be (float)H / outH and (float)W / outW
    float rheight = (outH > 1) ? (float)H / outH : 0.0f;
    float rwidth  = (outW > 1) ? (float)W / outW : 0.0f;

    upsample_bilinear2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, outH, outW, rheight, rwidth);
}

extern "C" void logical_and_range_launcher(
    const float* input,
    bool* output,
    int total,
    cudaStream_t stream)
{
    if (total == 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    logical_and_range_kernel<<<blocks, threads, 0, stream>>>(input, output, total);
}
