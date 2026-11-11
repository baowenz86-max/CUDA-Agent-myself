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
        #pragma unroll
        for (int kh = 0; kh < kernel_size; ++kh) {
            int64_t ih = ih_start + kh;
            if (ih >= in_height) continue;
            #pragma unroll
            for (int kw = 0; kw < kernel_size; ++kw) {
                int64_t iw = iw_start + kw;
                if (iw >= in_width) continue;
                int64_t offset = ih * in_stride_height + iw * in_stride_width;
                sum += base[offset];
            }
        }

        // 对于边缘窗口，除数为实际参与计算的元素个数
        int valid_h = min(kernel_size, (int)(in_height - ih_start));
        int valid_w = min(kernel_size, (int)(in_width - iw_start));
        int divisor = valid_h * valid_w;
        if (divisor > 0) {
            output[linear] = sum / divisor;
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
    int64_t spatial_size,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_height,
    int64_t in_stride_width)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;

    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        // 从线性索引解析出 n, c, h, w（基于输入 strides，假设输入形状为 [N, C, H, W]）
        // 这里输入可能非连续，使用与输入相同的 strides
        int64_t rem = linear;
        int64_t w = rem % in_stride_width; // 注意：in_stride_width 可能不为 1，需要谨慎
        // 上述方法仅在 strides 为常规时有效：in_stride_width = 1, in_stride_height = W, in_stride_channel = H*W, in_stride_batch = C*H*W
        // 但为了通用性，我们使用逐维除法，基于实际 strides。
        // 因为 binding 层会确保 strides 正确，此处采用公式。
        // 假设 in_stride_width = 1, in_stride_height = W, in_stride_channel = H*W, in_stride_batch = C*H*W
        // 更通用的做法是传递 width, height, batch 等，此处简化假设输入是连续的（在 binding 层保证）。
        // 实际上 custom_ops_contract 表示 add_bias 输入 supports strides，因此需要正确解析索引。
        // 我们传递 total_elements 表示按连续视图的长度，但输入可能非连续，因此需要谨慎。
        // 这里采用从线性索引映射到多维坐标，但需要 shape 信息。我们没有传递 shape，只传了 channels 和 spatial_size。
        // 为安全，假设输入是连续存储的 NCHW（binding 层会先做 contiguous），因此 strides 为 W, H*W, C*H*W。
        // 此实现目前按连续假设。
        int64_t w = linear % in_stride_height; // 因为 in_stride_height 是 W，in_stride_width=1
        int64_t h = (linear / in_stride_height) % (spatial_size / in_stride_height);
        // 更简单：binding 层确保传入连续张量和正确 strides，但此处我们可能忽略。
        // 为简化，我们传递必要的维度信息。
    }
}

// 上述 add_bias_kernel 不完整，需要修正。重新实现正确的内核。

__global__ void add_bias_kernel_v2(
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

    add_bias_kernel_v2<<<(unsigned int)blocks, block_size, 0, stream>>>(
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
