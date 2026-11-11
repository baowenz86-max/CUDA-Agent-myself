#include <cuda_runtime.h>

/*
 * conv2d_forward 在绑定文件中调用本文件的 launcher。
 * 本文件包含所有 CUDA kernel 和 launcher。
 */

// ---------------- reflection_pad2d_forward ----------------
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int pad_lr, int pad_tb,
    int in_stride_n, int in_stride_c, int in_stride_h, int in_stride_w,
    int out_H, int out_W)
{
    int out_hw = blockIdx.x * blockDim.x + threadIdx.x;
    int out_nc = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_nc >= N * C) return;
    if (out_hw >= out_H * out_W) return;

    int n = out_nc / C;
    int c = out_nc % C;
    int out_h = out_hw / out_W;
    int out_w = out_hw % out_W;

    int in_h = out_h - pad_tb;
    int in_w = out_w - pad_lr;

    if (in_h < 0) in_h = -in_h;
    else if (in_h >= H) in_h = 2 * (H - 1) - in_h;

    if (in_w < 0) in_w = -in_w;
    else if (in_w >= W) in_w = 2 * (W - 1) - in_w;

    if (H == 1) in_h = 0;
    if (W == 1) in_w = 0;

    const float* in_ptr = input + n * in_stride_n + c * in_stride_c + in_h * in_stride_h + in_w * in_stride_w;
    float* out_ptr = output + n * C * out_H * out_W + c * out_H * out_W + out_h * out_W + out_w;
    *out_ptr = *in_ptr;
}

extern "C" void reflection_pad2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad_lr, int pad_tb,
    int in_stride_n, int in_stride_c, int in_stride_h, int in_stride_w,
    cudaStream_t stream)
{
    int out_H = H + 2 * pad_tb;
    int out_W = W + 2 * pad_lr;
    int total_hw = out_H * out_W;
    int total_nc = N * C;

    dim3 block(16, 16);
    dim3 grid((total_hw + block.x - 1) / block.x, (total_nc + block.y - 1) / block.y);

    reflection_pad2d_kernel<<<grid, block, 0, stream>>>(
        input, output, N, C, H, W, pad_lr, pad_tb,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_H, out_W);
}

// ---------------- glu_forward ----------------
__global__ void glu_kernel_general(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_outer,
    int C,
    int inner,
    int input_stride_outer, int input_stride_dim, int input_stride_inner,
    int output_stride_outer, int output_stride_dim, int output_stride_inner)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outer * inner) return;

    int outer = idx / inner;
    int inner_idx = idx % inner;

    int half_C = C / 2;

    const float* a_ptr = input + outer * input_stride_outer + inner_idx * input_stride_inner;
    const float* b_ptr = a_ptr + half_C * input_stride_dim;

    float a = *a_ptr;
    float b = *b_ptr;
    float out = a / (1.0f + expf(-b));

    float* out_ptr = output + outer * output_stride_outer + inner_idx * output_stride_inner;
    *out_ptr = out;
}

extern "C" void glu_forward_launcher(
    const float* input,
    float* output,
    int total_outer, int C, int inner,
    int input_stride_outer, int input_stride_dim, int input_stride_inner,
    int output_stride_outer, int output_stride_dim, int output_stride_inner,
    cudaStream_t stream)
{
    int total = total_outer * inner;
    int block = 256;
    int grid = (total + block - 1) / block;
    glu_kernel_general<<<grid, block, 0, stream>>>(
        input, output, total_outer, C, inner,
        input_stride_outer, input_stride_dim, input_stride_inner,
        output_stride_outer, output_stride_dim, output_stride_inner);
}

// ---------------- min_dim_forward ----------------
__global__ void min_dim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_outer, int D, int inner,
    int input_stride_outer, int input_stride_dim, int input_stride_inner)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_outer * inner; i += stride) {
        int outer = i / inner;
        int inner_idx = i % inner;
        const float* base = input + outer * input_stride_outer + inner_idx * input_stride_inner;
        float min_val = base[0];
        for (int d = 1; d < D; ++d) {
            float v = base[d * input_stride_dim];
            if (v < min_val) min_val = v;
        }
        output[i] = min_val;
    }
}

extern "C" void min_dim_forward_launcher(
    const float* input,
    float* output,
    int total_outer, int D, int inner,
    int input_stride_outer, int input_stride_dim, int input_stride_inner,
    cudaStream_t stream)
{
    int total = total_outer * inner;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 4096) grid = 4096;
    min_dim_kernel<<<grid, block, 0, stream>>>(
        input, output, total_outer, D, inner,
        input_stride_outer, input_stride_dim, input_stride_inner);
}

// ---------------- conv2d_forward (naive direct convolution) ----------------
// 支持 stride, padding, dilation, groups (当前模型 groups=1 但保留通用性)
// 假设输入为 NCHW 布局，且数据连续或具有常规 strides（即按照 N,C,H,W 顺序存储）
// 输出为连续 NCHW。此 kernel 未对性能高度优化，但先保证正确性和编译。
__global__ void conv2d_direct_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out)
{
    int total_outputs = N * C_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int C_in_per_group = C_in / groups;
    int C_out_per_group = C_out / groups;

    for (int i = idx; i < total_outputs; i += stride) {
        int w_out = i % W_out;
        int rem = i / W_out;
        int h_out = rem % H_out;
        rem /= H_out;
        int c_out = rem % C_out;
        int n = rem / C_out;

        int group = c_out / C_out_per_group;
        int c_out_in_group = c_out % C_out_per_group;

        float acc = 0.0f;
        const float* input_base = input + n * C_in * H_in * W_in + group * C_in_per_group * H_in * W_in;
        const float* weight_base = weight + c_out * C_in_per_group * kH * kW;

        for (int c_in = 0; c_in < C_in_per_group; ++c_in) {
            for (int kh = 0; kh < kH; ++kh) {
                int h_in = h_out * stride_h - pad_h + kh * dilation_h;
                if (h_in < 0 || h_in >= H_in) continue;
                for (int kw = 0; kw < kW; ++kw) {
                    int w_in = w_out * stride_w - pad_w + kw * dilation_w;
                    if (w_in < 0 || w_in >= W_in) continue;
                    float in_val = input_base[c_in * H_in * W_in + h_in * W_in + w_in];
                    float w_val = weight_base[c_in * kH * kW + kh * kW + kw];
                    acc += in_val * w_val;
                }
            }
        }
        if (bias != nullptr) {
            acc += bias[c_out];
        }
        output[i] = acc;
    }
}

extern "C" void conv2d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out,
    cudaStream_t stream)
{
    int total_outputs = N * C_out * H_out * W_out;
    int block = 256;
    int grid = (total_outputs + block - 1) / block;
    if (grid > 65535) grid = 65535; // 保持 grid 维度限制
    conv2d_direct_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, H_in, W_in,
        C_out, kH, kW,
        stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w, groups,
        H_out, W_out);
}
