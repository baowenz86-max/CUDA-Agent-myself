#include <cuda_runtime.h>

// 双线性上采样 kernel：支持任意 stride，输出连续
// 对齐 corners=False: src = (dst + 0.5) / scale - 0.5
template <typename scalar_t>
__global__ void upsample_bilinear2d_cuda_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    float scale_h, float scale_w,
    int64_t in_N_stride, int64_t in_C_stride,
    int64_t in_H_stride, int64_t in_W_stride)
{
    int64_t total = (int64_t)N * C * H_out * W_out;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x)
    {
        int w_out = idx % W_out;
        int64_t tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        // 计算源坐标（align_corners=False）
        float src_h_f = (h_out + 0.5f) / scale_h - 0.5f;
        float src_w_f = (w_out + 0.5f) / scale_w - 0.5f;

        // clamp 到有效范围
        if (src_h_f < 0.0f) src_h_f = 0.0f;
        if (src_w_f < 0.0f) src_w_f = 0.0f;
        if (src_h_f > H_in - 1.0f) src_h_f = (float)(H_in - 1);
        if (src_w_f > W_in - 1.0f) src_w_f = (float)(W_in - 1);

        int h0 = (int)src_h_f;
        int w0 = (int)src_w_f;
        int h1 = (h0 < H_in - 1) ? (h0 + 1) : h0;
        int w1 = (w0 < W_in - 1) ? (w0 + 1) : w0;

        float h_lambda = src_h_f - h0;
        float w_lambda = src_w_f - w0;

        // 输入基地址（考虑 batch 和 channel 的 stride）
        const scalar_t* in_ptr = input + (int64_t)n * in_N_stride + (int64_t)c * in_C_stride;

        // 四个邻居值（使用给定 H,W stride）
        scalar_t v00 = in_ptr[(int64_t)h0 * in_H_stride + (int64_t)w0 * in_W_stride];
        scalar_t v01 = in_ptr[(int64_t)h0 * in_H_stride + (int64_t)w1 * in_W_stride];
        scalar_t v10 = in_ptr[(int64_t)h1 * in_H_stride + (int64_t)w0 * in_W_stride];
        scalar_t v11 = in_ptr[(int64_t)h1 * in_H_stride + (int64_t)w1 * in_W_stride];

        // 双线性插值
        float top = v00 * (1.0f - w_lambda) + v01 * w_lambda;
        float bottom = v10 * (1.0f - w_lambda) + v11 * w_lambda;
        float result = top * (1.0f - h_lambda) + bottom * h_lambda;

        output[idx] = (scalar_t)result;
    }
}

// 逻辑与 kernel: 输出 bool (uint8_t)
__global__ void logical_and_positive_less_than_one_kernel(
    const float* __restrict__ input,
    bool* __restrict__ output,
    int64_t total)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        float val = input[idx];
        output[idx] = (val > 0.0f) && (val < 1.0f);
    }
}

// Launchers (C interface, no torch dependencies)

extern "C" void upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    float scale_factor,
    int64_t in_N_stride, int64_t in_C_stride,
    int64_t in_H_stride, int64_t in_W_stride,
    cudaStream_t stream)
{
    int64_t total = (int64_t)N * C * H_out * W_out;
    if (total <= 0) return;

    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid cap, kernel uses grid-stride loop

    upsample_bilinear2d_cuda_kernel<float><<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, H_in, W_in, H_out, W_out,
        scale_factor, scale_factor,
        in_N_stride, in_C_stride, in_H_stride, in_W_stride);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // 错误处理：可以打印或忽略，但不能抛出异常（无 torch 头文件）
        // 这里简单记录，binding 层可通过返回检查
    }
}

extern "C" void logical_and_positive_less_than_one_launcher(
    const float* input,
    bool* output,
    int64_t total,
    cudaStream_t stream)
{
    if (total <= 0) return;

    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    logical_and_positive_less_than_one_kernel<<<blocks, threads, 0, stream>>>(
        input, output, total);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // 错误处理同上
    }
}
