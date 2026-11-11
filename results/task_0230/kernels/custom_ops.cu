// kernels/custom_ops.cu
// CUDA kernels for custom_selu_forward and custom_identity_forward.
// This file contains only CUDA kernels and launchers with raw pointers.

#include <cuda_runtime.h>
#include <cmath>

// SELU constants (double precision as in PyTorch, converted to float in kernel)
__constant__ float c_selu_alpha = 1.6732632423543772848170429916717f;
__constant__ float c_selu_scale = 1.0507009873554804934193349852946f;

// Kernel: elementwise SELU with arbitrary strides.
// Computes: out = scale * x if x > 0 else scale * alpha * (exp(x) - 1)
// Supports arbitrary strides via precomputed offset multipliers.
__global__ void selu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int numel,
    int batch_stride_in, int height_stride_in, int width_stride_in,
    int batch_stride_out, int height_stride_out, int width_stride_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < numel; i += stride) {
        // 将一维索引映射到 (b, h, w) 坐标
        int b = i / (height_stride_in * batch_stride_in == 0 ? 1 : height_stride_in * batch_stride_in); // 占位，实际使用下面的分解
        // 使用除法和取模：假设输入是三维，且 dim0 的 stride 乘以其大小等于总 numel 的倍数？
        // 更简单的方案：我们要求 binding 层传入的总 numel 和每个维度的 stride，并且假设输入是连续的或者至少是可以通过线性索引计算的。
        // 实际上，为了支持任意 stride，我们需要维度大小。但这里只有 stride 信息不够，还需要 shape。
        // 根据合同，shape 是固定的 [16, 5120, 5120]，所以我们可以从 numel 和 stride 推导 shape？
        // 但为了通用性，我们添加参数 shape0, shape1, shape2。
        // 我们将 launcher 设计为传入 shape 和 stride。
    }
}

// 由于上述索引计算需要 shape，重新定义 kernel 参数包含 shape 信息。
__global__ void selu_kernel_v2(
    float* __restrict__ output,
    const float* __restrict__ input,
    int numel,
    int shape0, int shape1, int shape2,
    long long stride0_in, long long stride1_in, long long stride2_in,
    long long stride0_out, long long stride1_out, long long stride2_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < numel; i += stride) {
        // 从一维索引重构 (b, h, w)
        int b = i / (shape1 * shape2);
        int rem = i - b * shape1 * shape2;
        int h = rem / shape2;
        int w = rem - h * shape2;

        long long in_offset = b * stride0_in + h * stride1_in + w * stride2_in;
        long long out_offset = b * stride0_out + h * stride1_out + w * stride2_out;

        float x = input[in_offset];
        float result;
        if (x > 0.0f) {
            result = c_selu_scale * x;
        } else {
            result = c_selu_scale * c_selu_alpha * (expf(x) - 1.0f);
        }
        output[out_offset] = result;
    }
}

// Identity kernel: simply copies input to output with arbitrary strides.
__global__ void identity_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int numel,
    int shape0, int shape1, int shape2,
    long long stride0_in, long long stride1_in, long long stride2_in,
    long long stride0_out, long long stride1_out, long long stride2_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < numel; i += stride) {
        int b = i / (shape1 * shape2);
        int rem = i - b * shape1 * shape2;
        int h = rem / shape2;
        int w = rem - h * shape2;

        long long in_offset = b * stride0_in + h * stride1_in + w * stride2_in;
        long long out_offset = b * stride0_out + h * stride1_out + w * stride2_out;

        output[out_offset] = input[in_offset];
    }
}

// Launcher for SELU operation.
extern "C" void custom_selu_forward_launcher(
    float* output,
    const float* input,
    int numel,
    int shape0, int shape1, int shape2,
    long long stride0_in, long long stride1_in, long long stride2_in,
    long long stride0_out, long long stride1_out, long long stride2_out,
    cudaStream_t stream)
{
    const int threads = 256;
    // 计算网格大小，限制最大网格尺寸。
    int blocks = (numel + threads - 1) / threads;
    const int max_blocks = 65535;
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }
    selu_kernel_v2<<<blocks, threads, 0, stream>>>(
        output, input, numel,
        shape0, shape1, shape2,
        stride0_in, stride1_in, stride2_in,
        stride0_out, stride1_out, stride2_out);
}

// Launcher for identity operation.
extern "C" void custom_identity_forward_launcher(
    float* output,
    const float* input,
    int numel,
    int shape0, int shape1, int shape2,
    long long stride0_in, long long stride1_in, long long stride2_in,
    long long stride0_out, long long stride1_out, long long stride2_out,
    cudaStream_t stream)
{
    const int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    const int max_blocks = 65535;
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }
    identity_kernel<<<blocks, threads, 0, stream>>>(
        output, input, numel,
        shape0, shape1, shape2,
        stride0_in, stride1_in, stride2_in,
        stride0_out, stride1_out, stride2_out);
}
