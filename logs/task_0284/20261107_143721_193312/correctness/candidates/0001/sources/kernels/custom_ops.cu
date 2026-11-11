#include <cuda_runtime.h>

/*
 * conv2d_forward 由第三阶段的 C++ 绑定文件使用 cuDNN 实现，不在本文件实现。
 * 本文件仅包含非卷积 CUDA kernel 和 launcher。
 */

// ---------------- reflection_pad2d_forward ----------------
// 输入: input (N, C, H, W), pad_left_right, pad_top_bottom
// 输出: output (N, C, H+2*pad_top_bottom, W+2*pad_left_right)
// 支持任意 strides，输出连续。
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

    // 反射索引计算，不重复边缘值
    if (in_h < 0) in_h = -in_h;
    else if (in_h >= H) in_h = 2 * (H - 1) - in_h;

    if (in_w < 0) in_w = -in_w;
    else if (in_w >= W) in_w = 2 * (W - 1) - in_w;

    // 边界情况：当 H=1 时，反射索引可能仍为0，公式仍成立。
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
// 输入: input (..., C, ...) 连续? 支持 strides
// dim: 分割维度
// 输出: 形状与输入相同但 dim 维度减半，连续。
// 该实现将输入视为任意维度，但使用 strides 进行索引。
// 由于 GLU 通常用于4维，且 dim=1，这里实现通用但针对性能优化为4维 dim=1 情况，也支持其他情况。
// 注意：输出为连续，因此可以利用线性索引，但为简单起见使用嵌套循环。
__global__ void glu_kernel_general(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total_outer, // 外层循环长度
    int C, // dim 维度大小
    int inner, // 内层循环长度
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
    float out = a / (1.0f + expf(-b)); // sigmoid(b) = 1/(1+exp(-b))

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
// 输入: input 任意维，dim 指定维度。
// 输出: 形状与输入相同但 dim 维度移除，连续。
// 实现为每个输出元素独立计算，使用 grid-stride loop。
// 假设输出连续，但支持输入任意 strides。
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
    // 限制 grid 大小，使用 grid-stride loop
    if (grid > 4096) grid = 4096;
    min_dim_kernel<<<grid, block, 0, stream>>>(
        input, output, total_outer, D, inner,
        input_stride_outer, input_stride_dim, input_stride_inner);
}
