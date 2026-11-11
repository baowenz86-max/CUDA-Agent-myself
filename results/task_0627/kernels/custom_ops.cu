#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// pad_sum_channels: fused ZeroPad2d + sum(dim=1, keepdim=True)
// input:  x [B, C, H, W] float32 (strided but assumed contiguous for simplicity
//         since PyTorch tensors from randn are contiguous and pad_sum_channels
//         binding will call .contiguous() before passing)
// output: out [B, 1, H+top+bottom, W+left+right] float32
// -----------------------------------------------------------------------------
__global__ void pad_sum_channels_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int C, int H, int W,
    int left, int right, int top, int bottom,
    int outH, int outW)
{
    int total = B * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int ow = i % outW;
        int oh = (i / outW) % outH;
        int b = i / (outW * outH);
        float sum = 0.0f;
        int ih = oh - top;
        int iw = ow - left;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            const float* base = x + ((long long)b * C) * (H * W) + ih * W + iw;
            for (int c = 0; c < C; ++c) {
                sum += base[(long long)c * (H * W)];
            }
        }
        out[i] = sum;
    }
}

extern "C" void pad_sum_channels_launcher(
    const float* x,
    float* out,
    int B, int C, int H, int W,
    int left, int right, int top, int bottom,
    cudaStream_t stream)
{
    int outH = H + top + bottom;
    int outW = W + left + right;
    int total = B * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    pad_sum_channels_kernel<<<blocks, threads, 0, stream>>>(
        x, out, B, C, H, W, left, right, top, bottom, outH, outW);
}

// -----------------------------------------------------------------------------
// pad2d_custom: ZeroPad2d only
// input:  x [B, C, H, W] float32
// output: out [B, C, H+top+bottom, W+left+right] float32, zeros padded
// -----------------------------------------------------------------------------
__global__ void pad2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int C, int H, int W,
    int left, int right, int top, int bottom,
    int outH, int outW)
{
    int total = B * C * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int ow = i % outW;
        int oh = (i / outW) % outH;
        int c = (i / (outW * outH)) % C;
        int b = i / (outW * outH * C);
        int ih = oh - top;
        int iw = ow - left;
        float val = 0.0f;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            val = x[((long long)b * C + c) * (H * W) + ih * W + iw];
        }
        out[i] = val;
    }
}

extern "C" void pad2d_launcher(
    const float* x,
    float* out,
    int B, int C, int H, int W,
    int left, int right, int top, int bottom,
    cudaStream_t stream)
{
    int outH = H + top + bottom;
    int outW = W + left + right;
    int total = B * C * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    pad2d_kernel<<<blocks, threads, 0, stream>>>(
        x, out, B, C, H, W, left, right, top, bottom, outH, outW);
}

// -----------------------------------------------------------------------------
// sum_channels_custom: sum over channel dimension (keepdim=true)
// input:  x [B, C, H, W] float32
// output: out [B, 1, H, W] float32
// -----------------------------------------------------------------------------
__global__ void sum_channels_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int C, int H, int W)
{
    int total = B * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int w = i % W;
        int h = (i / W) % H;
        int b = i / (H * W);
        float sum = 0.0f;
        const float* base = x + ((long long)b * C) * (H * W) + h * W + w;
        for (int c = 0; c < C; ++c) {
            sum += base[(long long)c * (H * W)];
        }
        out[i] = sum;
    }
}

extern "C" void sum_channels_launcher(
    const float* x,
    float* out,
    int B, int C, int H, int W,
    cudaStream_t stream)
{
    int total = B * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    sum_channels_kernel<<<blocks, threads, 0, stream>>>(
        x, out, B, C, H, W);
}
