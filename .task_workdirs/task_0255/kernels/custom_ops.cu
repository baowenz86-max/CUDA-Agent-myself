#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// avgpool2d_forward
// 对应 reference_steps step_001: x = self.avg_pool(x)
// AvgPool2d kernel_size=3, stride=3, padding=0
// 输入支持任意 strides，输出为连续 NCHW 张量。
// -----------------------------------------------------------------------------

__global__ void avgpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_elements,
    int64_t batch,
    int64_t channels,
    int64_t in_height,
    int64_t in_width,
    int64_t out_height,
    int64_t out_width,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_height,
    int64_t in_stride_width)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        // 将 linear 索引映射到输出坐标 (n, c, oh, ow)
        int64_t ow = linear % out_width;
        int64_t tmp = linear / out_width;
        int64_t oh = tmp % out_height;
        tmp /= out_height;
        int64_t c = tmp % channels;
        int64_t n = tmp / channels;

        const int kernel_size = 3;
        const int stride_pool = 3;

        // 计算输入窗口起始位置（无 padding）
        int64_t ih_start = oh * stride_pool;
        int64_t iw_start = ow * stride_pool;

        // 输入基指针
        const float* base = input + n * in_stride_batch + c * in_stride_channel;

        float sum = 0.0f;
        int count = 0;
        for (int kh = 0; kh < kernel_size; ++kh) {
            int64_t ih = ih_start + kh;
            if (ih >= in_height) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int64_t iw = iw_start + kw;
                if (iw >= in_width) continue;
                int64_t offset = ih * in_stride_height + iw * in_stride_width;
                sum += base[offset];
                ++count;
            }
        }

        // 对于边缘窗口，除数为实际参与计算的元素个数
        if (count > 0) {
            output[linear] = sum / count;
        } else {
            output[linear] = 0.0f; // 不应发生
        }
    }
}

extern "C" void avgpool2d_forward_launcher(
    const float* input,
    float* output,
    int64_t batch,
    int64_t channels,
    int64_t in_height,
    int64_t in_width,
    int64_t out_height,
    int64_t out_width,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_height,
    int64_t in_stride_width,
    cudaStream_t stream)
{
    int64_t total_elements = batch * channels * out_height * out_width;
    if (total_elements <= 0) return;

    const int block_size = 256;
    int64_t blocks = (total_elements + block_size - 1) / block_size;
    // 限制 grid 大小，使用 grid-stride loop
    if (blocks > 65535) blocks = 65535;

    avgpool2d_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
        input,
        output,
        total_elements,
        batch,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        in_stride_batch,
        in_stride_channel,
        in_stride_height,
        in_stride_width);
}

// -----------------------------------------------------------------------------
// conv_transpose2d CUDA kernel (无 bias)
// 实现 ConvTranspose2d 前向，参数：stride=2, padding=1, output_padding=1, kernel=3
// 输入形状 [N, C_in, H_in, W_in]，权重形状 [C_in, C_out, kH, kW]
// 输出形状 [N, C_out, H_out, W_out]
// 采用直接计算：对每个输出位置，累加所有输入位置与对应权重的乘积
// -----------------------------------------------------------------------------

__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t total_elements,
    int64_t N,
    int64_t C_in,
    int64_t C_out,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int64_t kH,
    int64_t kW,
    int64_t stride,
    int64_t padding,
    int64_t output_padding)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride_grid = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t linear = idx; linear < total_elements; linear += stride_grid) {
        // 解析输出坐标 (n, co, oh, ow)
        int64_t ow = linear % W_out;
        int64_t tmp = linear / W_out;
        int64_t oh = tmp % H_out;
        tmp /= H_out;
        int64_t co = tmp % C_out;
        int64_t n = tmp / C_out;

        float sum = 0.0f;

        // 遍历输入通道和空间位置
        for (int ci = 0; ci < C_in; ++ci) {
            const float* input_base = input + n * C_in * H_in * W_in + ci * H_in * W_in;
            const float* weight_base = weight + ci * C_out * kH * kW + co * kH * kW;

            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    // 计算对应的输入位置：
                    // oh = ih * stride - padding + kh
                    // ow = iw * stride - padding + kw
                    int ih = oh + padding - kh;
                    int iw = ow + padding - kw;
                    if (ih % stride == 0 && iw % stride == 0) {
                        int ih_actual = ih / stride;
                        int iw_actual = iw / stride;
                        if (ih_actual >= 0 && ih_actual < H_in && iw_actual >= 0 && iw_actual < W_in) {
                            sum += input_base[ih_actual * W_in + iw_actual] * weight_base[kh * kW + kw];
                        }
                    }
                }
            }
        }

        output[linear] = sum;
    }
}

extern "C" void conv_transpose2d_forward_launcher(
    const float* input,
    const float* weight,
    float* output,
    int64_t N,
    int64_t C_in,
    int64_t C_out,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int64_t kH,
    int64_t kW,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    cudaStream_t stream)
{
    int64_t total_elements = N * C_out * H_out * W_out;
    if (total_elements <= 0) return;

    const int block_size = 256;
    int64_t blocks = (total_elements + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;

    conv_transpose2d_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
        input,
        weight,
        output,
        total_elements,
        N,
        C_in,
        C_out,
        H_in,
        W_in,
        H_out,
        W_out,
        kH,
        kW,
        stride,
        padding,
        output_padding);
}

// -----------------------------------------------------------------------------
// add_bias_forward
// 对应 reference_steps step_003: x = x + self.bias
// 将 bias 沿通道广播，形状 [out_channels, 1, 1]
// 输出与输入具有相同的 shape 与 strides（binding 层分配相同 strides）。
// -----------------------------------------------------------------------------

__global__ void add_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t total_elements,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_height,
    int64_t in_stride_width)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        // 显式计算 n, c, h, w，使用 shape 信息和 strides
        int64_t w = linear % width;
        int64_t tmp = linear / width;
        int64_t h = tmp % height;
        tmp /= height;
        int64_t c = tmp % channels;
        int64_t n = tmp / channels;

        int64_t input_offset = n * in_stride_batch + c * in_stride_channel + h * in_stride_height + w * in_stride_width;
        int64_t output_offset = input_offset; // 输出使用与输入相同的 strides

        output[output_offset] = input[input_offset] + bias[c];
    }
}

extern "C" void add_bias_forward_launcher(
    const float* input,
    const float* bias,
    float* output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_height,
    int64_t in_stride_width,
    cudaStream_t stream)
{
    int64_t total_elements = batch * channels * height * width;
    if (total_elements <= 0) return;

    const int block_size = 256;
    int64_t blocks = (total_elements + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;

    add_bias_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
        input,
        bias,
        output,
        total_elements,
        channels,
        height,
        width,
        in_stride_batch,
        in_stride_channel,
        in_stride_height,
        in_stride_width);
}
