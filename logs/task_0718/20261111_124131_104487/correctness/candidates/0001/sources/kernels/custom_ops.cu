#include <cuda_runtime.h>
#include <cstdint>

// custom_sum_dim1: reduce tensor x of shape [B, C, H, W] along dimension 1.
// Output shape [B, H, W]. Supports arbitrary strides.
__global__ void custom_sum_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t stride_b,
    int64_t stride_c,
    int64_t stride_h,
    int64_t stride_w,
    int64_t out_stride_b,
    int64_t out_stride_h,
    int64_t out_stride_w)
{
    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_out = B * H * W;
    if (out_idx >= total_out) return;

    // decompose output index to (b, h, w)
    int64_t b = out_idx / (H * W);
    int64_t rem = out_idx - b * (H * W);
    int64_t h = rem / W;
    int64_t w = rem - h * W;

    float sum = 0.0f;
    const float* base = x + b * stride_b + h * stride_h + w * stride_w;
    for (int64_t c = 0; c < C; ++c) {
        sum += base[c * stride_c];
    }

    out[b * out_stride_b + h * out_stride_h + w * out_stride_w] = sum;
}

// custom_fmod: elementwise fmod of input tensor of shape [N] with divisor.
// Supports arbitrary strides.
__global__ void custom_fmod_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t n,
    int64_t in_stride,
    int64_t out_stride,
    float divisor)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = x[idx * in_stride];
    out[idx * out_stride] = fmodf(val, divisor);
}

// Launcher for custom_sum_dim1
// x: input tensor [B,C,H,W] (contiguous or not)
// out: output tensor [B,H,W]
// strides: input strides for b,c,h,w; output strides for b,h,w
// stream: CUDA stream
void custom_sum_dim1_launcher(
    const float* x,
    float* out,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t stride_b,
    int64_t stride_c,
    int64_t stride_h,
    int64_t stride_w,
    int64_t out_stride_b,
    int64_t out_stride_h,
    int64_t out_stride_w,
    cudaStream_t stream)
{
    int64_t total_out = B * H * W;
    int block_size = 256;
    int grid_size = (int)((total_out + block_size - 1) / block_size);
    custom_sum_dim1_kernel<<<grid_size, block_size, 0, stream>>>(
        x, out, B, C, H, W,
        stride_b, stride_c, stride_h, stride_w,
        out_stride_b, out_stride_h, out_stride_w);
}

// Launcher for custom_fmod
// x: input tensor [N] (contiguous or not)
// out: output tensor [N]
// divisor: scalar divisor (double)
// stream: CUDA stream
void custom_fmod_launcher(
    const float* x,
    float* out,
    int64_t n,
    int64_t in_stride,
    int64_t out_stride,
    double divisor,
    cudaStream_t stream)
{
    // C++ standard fmodf is not available in device code? Actually fmodf is supported.
    // Cast divisor to float for device.
    float fdivisor = (float)divisor;
    int block_size = 256;
    int grid_size = (int)((n + block_size - 1) / block_size);
    custom_fmod_kernel<<<grid_size, block_size, 0, stream>>>(
        x, out, n, in_stride, out_stride, fdivisor);
}
