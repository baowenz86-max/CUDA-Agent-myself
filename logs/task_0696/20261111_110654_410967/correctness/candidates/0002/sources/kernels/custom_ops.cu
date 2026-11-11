#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// copy_to_cuda_forward: 将 CPU 上的 float32 连续张量复制到当前 CUDA 设备。
// 输入: src_cpu 指向 CPU 内存; numel 元素个数; stream 当前 CUDA stream。
// 输出: dst_cuda 指向 GPU 内存（由 binding 层预先分配并传入）。
// 该 launcher 不负责分配内存，只执行复制。
// -----------------------------------------------------------------------------
extern "C" void copy_to_cuda_launcher(
    float* dst_cuda,
    const float* src_cpu,
    int64_t numel,
    cudaStream_t stream)
{
    cudaError_t err = cudaMemcpyAsync(dst_cuda, src_cpu, numel * sizeof(float), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        // 错误由调用者检查，这里不做处理
    }
}

// -----------------------------------------------------------------------------
// pad_forward: 对输入四维张量 [N, C, H, W] 在 H 和 W 维度各填充 pad 个零。
// 输入: src 连续 GPU float32 张量, N C H W, pad 填充宽度(此处固定为2)。
// 输出: dst 连续 GPU float32 张量，形状 [N, C, H+2*pad, W+2*pad]，已清零。
// 使用 grid-stride loop，每个线程负责一个输出元素，直接复制或写零。
// -----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void pad_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t pad)
{
    int64_t out_H = H + 2 * pad;
    int64_t out_W = W + 2 * pad;
    int64_t total_out = N * C * out_H * out_W;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_out;
         idx += gridDim.x * blockDim.x) {

        // 将线性索引分解为输出坐标 (n, c, oh, ow)
        int64_t ow = idx % out_W;
        int64_t tmp = idx / out_W;
        int64_t oh = tmp % out_H;
        tmp /= out_H;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // 检查是否在填充区域内
        if (oh >= pad && oh < H + pad && ow >= pad && ow < W + pad) {
            // 有效数据区域，复制输入对应元素
            int64_t ih = oh - pad;
            int64_t iw = ow - pad;
            int64_t src_idx = ((n * C + c) * H + ih) * W + iw;
            dst[idx] = src[src_idx];
        } else {
            // 填充区域，置零
            dst[idx] = 0.0f;
        }
    }
}

extern "C" void pad_launcher(
    const float* src,
    float* dst,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t pad,
    cudaStream_t stream)
{
    const int block_size = 256;
    int64_t out_H = H + 2 * pad;
    int64_t out_W = W + 2 * pad;
    int64_t total_out = N * C * out_H * out_W;
    // 计算足够的 block 数量，限制最大 grid 维度
    int64_t blocks64 = (total_out + block_size - 1) / block_size;
    int blocks = static_cast<int>(min(blocks64, static_cast<int64_t>(INT32_MAX)));
    if (blocks <= 0) blocks = 1;

    pad_kernel<256><<<blocks, block_size, 0, stream>>>(
        src, dst, N, C, H, W, pad);
}

// -----------------------------------------------------------------------------
// sum_dim_forward: 沿指定维度 dim=2 (height) 求和。
// 输入: src 连续 GPU float32 张量 [N, C, H, W]。
// 输出: dst 连续 GPU float32 张量 [N, C, W]，初始化为零。
// 为简单和正确，每个输出元素由一个线程累加整个 H 维度。
// 使用 grid-stride 循环覆盖所有输出元素。
// -----------------------------------------------------------------------------
__global__ void sum_dim2_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int64_t N, int64_t C, int64_t H, int64_t W)
{
    int64_t total_out = N * C * W;

    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_out;
         idx += gridDim.x * blockDim.x) {

        // 分解输出索引 (n, c, w)
        int64_t w = idx % W;
        int64_t tmp = idx / W;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // 累加 H 维度
        float sum = 0.0f;
        for (int64_t h = 0; h < H; ++h) {
            int64_t src_idx = ((n * C + c) * H + h) * W + w;
            sum += src[src_idx];
        }
        dst[idx] = sum;
    }
}

extern "C" void sum_dim_launcher(
    const float* src,
    float* dst,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t dim,
    cudaStream_t stream)
{
    // 仅支持 dim=2，其他 dim 由绑定层拒绝或使用通用实现（此处未实现）
    if (dim != 2) {
        return;  // 调用者应处理错误
    }

    const int block_size = 256;
    int64_t total_out = N * C * W;
    int64_t blocks64 = (total_out + block_size - 1) / block_size;
    int blocks = static_cast<int>(min(blocks64, static_cast<int64_t>(INT32_MAX)));
    if (blocks <= 0) blocks = 1;

    sum_dim2_kernel<<<blocks, block_size, 0, stream>>>(
        src, dst, N, C, H, W);
}
