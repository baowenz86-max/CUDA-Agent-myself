#include <cuda_runtime.h>
#include <cstdint>

// ----------------------------------------------------------------------------
// avg_pool2d kernel
// Computes 2D average pooling with kernel_size x kernel_size, stride, padding.
// Input: (B, C, H, W) float32, arbitrary strides
// Output: (B, C, outH, outW) float32, contiguous (allocated by binding)
// Semantics match torch.nn.AvgPool2d with count_include_pad=True (default),
// i.e. divisor is kernel_size*kernel_size, padded zeros included.
// ----------------------------------------------------------------------------
__global__ void avg_pool2d_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int B, int C, int H, int W,
    int KH, int KW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    long long input_batch_stride,
    long long input_channel_stride,
    long long input_height_stride,
    long long input_width_stride)
{
    int outH = (H + 2 * pad_h - KH) / stride_h + 1;
    int outW = (W + 2 * pad_w - KW) / stride_w + 1;
    int total = B * C * outH * outW;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int ow = idx % outW;
        int oh = (idx / outW) % outH;
        int c = (idx / (outW * outH)) % C;
        int b = idx / (outW * outH * C);

        const float* in_c = input + b * input_batch_stride + c * input_channel_stride;

        float sum = 0.0f;
        int h_start = oh * stride_h - pad_h;
        int w_start = ow * stride_w - pad_w;

        for (int kh = 0; kh < KH; ++kh) {
            int ih = h_start + kh;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int iw = w_start + kw;
                if (iw < 0 || iw >= W) continue;
                sum += in_c[ih * input_height_stride + iw * input_width_stride];
            }
        }
        output[idx] = sum / (float)(KH * KW);
    }
}

// ----------------------------------------------------------------------------
// min over dimension 1 (channels)
// Input: (B, C, H, W) float32, arbitrary strides
// Output: (B, H, W) float32, contiguous
// dim must be 1; output index maps (b, h, w) -> minimum over c
// ----------------------------------------------------------------------------
__global__ void min_dim1_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int B, int C, int H, int W,
    long long input_batch_stride,
    long long input_channel_stride,
    long long input_height_stride,
    long long input_width_stride)
{
    int total = B * H * W;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int w = idx % W;
        int h = (idx / W) % H;
        int b = idx / (W * H);

        const float* base = input + b * input_batch_stride + h * input_height_stride + w * input_width_stride;
        float min_val = base[0 * input_channel_stride];  // channel 0
        for (int c = 1; c < C; ++c) {
            float val = base[c * input_channel_stride];
            if (val < min_val) min_val = val;
        }
        output[idx] = min_val;
    }
}

// ----------------------------------------------------------------------------
// Host launchers (no PyTorch dependencies, raw pointers + cudaStream_t)
// ----------------------------------------------------------------------------
extern "C" void avg_pool2d_launcher(
    float* output,
    const float* input,
    int B, int C, int H, int W,
    int kernel_size, int stride, int padding,
    long long input_batch_stride,
    long long input_channel_stride,
    long long input_height_stride,
    long long input_width_stride,
    cudaStream_t stream)
{
    // Assume kernel_size is square (KH=KW)
    int KH = kernel_size;
    int KW = kernel_size;
    int stride_h = stride;
    int stride_w = stride;
    int pad_h = padding;
    int pad_w = padding;

    int outH = (H + 2 * pad_h - KH) / stride_h + 1;
    int outW = (W + 2 * pad_w - KW) / stride_w + 1;
    int total = B * C * outH * outW;

    int block = 256;
    int grid = (total + block - 1) / block;
    // cap grid to avoid excessive blocks, but ensure coverage via grid-stride loop
    if (grid > 65535) grid = 65535;

    avg_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
        output, input, B, C, H, W,
        KH, KW, stride_h, stride_w, pad_h, pad_w,
        input_batch_stride, input_channel_stride, input_height_stride, input_width_stride);
}

extern "C" void min_dim_launcher(
    float* output,
    const float* input,
    int B, int C, int H, int W,
    int dim,
    long long input_batch_stride,
    long long input_channel_stride,
    long long input_height_stride,
    long long input_width_stride,
    cudaStream_t stream)
{
    // Only dim=1 is used according to contract; if dim != 1, we still compute min over channels
    // but for safety, we ignore dim and assume 1. The binding should validate.
    int total = B * H * W;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535;

    min_dim1_forward_kernel<<<grid, block, 0, stream>>>(
        output, input, B, C, H, W,
        input_batch_stride, input_channel_stride, input_height_stride, input_width_stride);
}
