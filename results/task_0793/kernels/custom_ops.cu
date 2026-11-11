#include <cuda_runtime.h>

// isin kernel: 判断 x 的每个元素是否等于 elements 中任意值
__global__ void isin_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ elements,
    bool* __restrict__ output,
    long long total_elements,
    int elements_count)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += stride) {
        float val = x[idx];
        bool found = false;
        for (int i = 0; i < elements_count; ++i) {
            if (val == elements[i]) {
                found = true;
                break;
            }
        }
        output[idx] = found;
    }
}

// zero_pad2d kernel: 最后两个维度填充
// 输入: [N, C, H, W], 输出: [N, C, H + pad_top + pad_bottom, W + pad_left + pad_right]
__global__ void zero_pad2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    long long total_output_elements)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    int out_H = H + pad_top + pad_bottom;
    int out_W = W + pad_left + pad_right;
    for (; idx < total_output_elements; idx += stride) {
        // 将线性索引转换为 4D 坐标 (n, c, oh, ow)
        int ow = idx % out_W;
        long long tmp = idx / out_W;
        int oh = tmp % out_H;
        tmp /= out_H;
        int c = tmp % C;
        int n = tmp / C;

        int ih = oh - pad_top;
        int iw = ow - pad_left;

        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            // 复制输入元素
            long long input_idx = ((long long)n * C + c) * H * W + (long long)ih * W + iw;
            output[idx] = input[input_idx];
        } else {
            output[idx] = 0.0f;
        }
    }
}

// host launcher: isin_forward
extern "C" void isin_forward_launcher(
    const float* x,
    const float* elements,
    bool* output,
    long long total_elements,
    int elements_count,
    cudaStream_t stream)
{
    const int threads = 256;
    // 使用足够的块数，grid-stride loop 保证覆盖
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    isin_forward_kernel<<<blocks, threads, 0, stream>>>(
        x, elements, output, total_elements, elements_count);
}

// host launcher: zero_pad2d_forward
extern "C" void zero_pad2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    long long total_output_elements,
    cudaStream_t stream)
{
    const int threads = 256;
    int blocks = (int)((total_output_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    zero_pad2d_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W,
        pad_left, pad_right, pad_top, pad_bottom,
        total_output_elements);
}
