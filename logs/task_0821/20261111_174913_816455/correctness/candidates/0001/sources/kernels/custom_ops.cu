#include <cuda_runtime.h>
#include <cstdint>

// 反射填充索引计算：将索引映射到 [0, size)
__device__ __forceinline__ int reflect_index(int idx, int size) {
    if (size <= 1) return 0;
    // 使用取模反射，不重复边界
    int m = 2 * size - 2;
    idx = ((idx % m) + m) % m;
    if (idx >= size) idx = m - idx;
    return idx;
}

// 反射填充 kernel: 输入 [N, C, H, W]，输出 [N, C, OH, OW]
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int OH, int OW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * OH * OW;
    if (idx >= total) return;

    // 分解线性索引为 (n, c, oh, ow)
    int ow = idx % OW;
    int tmp = idx / OW;
    int oh = tmp % OH;
    tmp /= OH;
    int c = tmp % C;
    int n = tmp / C;

    // 计算源输入坐标
    int iw = reflect_index(ow - pad_left, W);
    int ih = reflect_index(oh - pad_top, H);

    int in_idx = ((n * C + c) * H + ih) * W + iw;
    int out_idx = ((n * C + c) * OH + oh) * OW + ow;
    output[out_idx] = input[in_idx];
}

// 沿通道维 argsort kernel: 输入 [N, C, H, W]，输出 [N, C, H, W] int64 索引
// 对每个 (n, h, w) 位置，对 C 个元素排序，升序。
__global__ void argsort_dim1_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    int N, int C, int H, int W)
{
    int hw_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int HW = H * W;
    if (hw_idx >= N * HW) return;

    int n = hw_idx / HW;
    int hw = hw_idx % HW;
    int h = hw / W;
    int w = hw % W;

    // 从输入中读取 C 个值到寄存器数组（C 通常很小，如3）
    // 这里使用固定最大通道数处理，避免动态共享内存开销
    float vals[8]; // 支持 C <= 8，本任务 C=3
    int indices[8];

    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        if (c < C) {
            int in_idx = ((n * C + c) * H + h) * W + w;
            vals[c] = input[in_idx];
            indices[c] = c;
        }
    }

    // 简单插入排序（通道数小时高效）
    #pragma unroll
    for (int i = 1; i < 8; ++i) {
        if (i < C) {
            float key_val = vals[i];
            int key_idx = indices[i];
            int j = i - 1;
            while (j >= 0 && vals[j] > key_val) {
                vals[j + 1] = vals[j];
                indices[j + 1] = indices[j];
                --j;
            }
            vals[j + 1] = key_val;
            indices[j + 1] = key_idx;
        }
    }

    // 写回输出
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        if (c < C) {
            int out_idx = ((n * C + c) * H + h) * W + w;
            output[out_idx] = indices[c];
        }
    }
}

// Launcher: reflection_pad2d_forward
// 参数: x (float*), N, C, H, W, pad_left, pad_right, pad_top, pad_bottom, output (float*), stream
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    cudaStream_t stream)
{
    int OH = H + pad_top + pad_bottom;
    int OW = W + pad_left + pad_right;
    int total = N * C * OH * OW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W,
        pad_left, pad_right, pad_top, pad_bottom,
        OH, OW);
}

// Launcher: argsort_dim1_forward
// 参数: x (float*), N, C, H, W, output (int64_t*), stream
extern "C" void argsort_dim1_launcher(
    const float* input,
    int64_t* output,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int HW = H * W;
    int total_positions = N * HW;
    int threads = 256;
    int blocks = (total_positions + threads - 1) / threads;

    argsort_dim1_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W);
}
