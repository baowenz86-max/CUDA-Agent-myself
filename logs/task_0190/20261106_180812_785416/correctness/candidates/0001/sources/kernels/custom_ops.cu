#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// prod_channel_keepdim kernel
// 计算 x 张量的通道维(dim=1)乘积，保持维度。
// 输入 x 形状: [N, C, H, W]，输出形状: [N, 1, H, W]
// 布局支持任意 stride。
// -----------------------------------------------------------------------------
__global__ void prod_channel_keepdim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    int hw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_hw = H * W;
    if (hw_idx >= total_hw) return;

    int h = hw_idx / W;
    int w = hw_idx - h * W;

    for (int n = 0; n < N; ++n) {
        const float* in_base = input + n * in_stride_n + h * in_stride_h + w * in_stride_w;
        float prod = in_base[0];
        for (int c = 1; c < C; ++c) {
            prod *= in_base[c * in_stride_c];
        }
        float* out_ptr = output + n * out_stride_n + h * out_stride_h + w * out_stride_w;
        *out_ptr = prod;
    }
}

extern "C" void prod_channel_keepdim_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    int total_hw = H * W;
    int block = 256;
    int grid = (total_hw + block - 1) / block;
    prod_channel_keepdim_kernel<<<grid, block, 0, stream>>>(
        input, output, N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}

// -----------------------------------------------------------------------------
// max_pool2d kernel
// 2D 最大池化，kernel_size=(kh, kw), stride=(sh, sw), padding=(ph, pw), dilation=1
// 输入形状: [N, C, H_in, W_in]，输出形状: [N, C, H_out, W_out]
// 支持任意 stride，使用 input_stride 访问。
// -----------------------------------------------------------------------------
__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    int kh, int kw, int sh, int sw, int ph, int pw,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;

    int h_start = h_out * sh - ph;
    int w_start = w_out * sw - pw;

    const float* in_base = input + n * in_stride_n + c * in_stride_c;
    float max_val = -3.402823466e+38f; // -FLT_MAX

    for (int i = 0; i < kh; ++i) {
        int h_in = h_start + i;
        if (h_in < 0 || h_in >= H_in) continue;
        for (int j = 0; j < kw; ++j) {
            int w_in = w_start + j;
            if (w_in < 0 || w_in >= W_in) continue;
            float val = in_base[h_in * in_stride_h + w_in * in_stride_w];
            if (val > max_val) max_val = val;
        }
    }

    float* out_ptr = output + n * out_stride_n + c * out_stride_c + h_out * out_stride_h + w_out * out_stride_w;
    *out_ptr = max_val;
}

extern "C" void max_pool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    int kh, int kw, int sh, int sw, int ph, int pw,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    int total = N * C * H_out * W_out;
    int block = 256;
    int grid = (total + block - 1) / block;
    max_pool2d_kernel<<<grid, block, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out,
        kh, kw, sh, sw, ph, pw,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}
