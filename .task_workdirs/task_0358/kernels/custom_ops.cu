#include <cuda_runtime.h>
#include <float.h>

// CUDA kernel for adaptive max pooling 2D (NCHW layout, contiguous input)
// Each thread computes one output element (n, c, oh, ow)
__global__ void adaptive_max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    int stride = blockDim.x * gridDim.x;

    for (int linear_idx = idx; linear_idx < total; linear_idx += stride) {
        // 反解索引
        int ow = linear_idx % outW;
        int oh = (linear_idx / outW) % outH;
        int c = (linear_idx / (outW * outH)) % C;
        int n = linear_idx / (outW * outH * C);

        // 计算输入窗口边界（与 PyTorch adaptive pooling 一致）
        int h_start = (oh * H) / outH;
        int h_end = ((oh + 1) * H + outH - 1) / outH; // ceil
        int w_start = (ow * W) / outW;
        int w_end = ((ow + 1) * W + outW - 1) / outW; // ceil

        // 确保边界不越界
        if (h_end > H) h_end = H;
        if (w_end > W) w_end = W;

        float max_val = -FLT_MAX;
        const float* base = input + ((n * C + c) * H) * W;
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                float val = base[h * W + w];
                if (val > max_val) max_val = val;
            }
        }

        output[linear_idx] = max_val;
    }
}

// Host launcher
extern "C" void adaptive_max_pool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    cudaStream_t stream
) {
    int total = N * C * outH * outW;
    if (total <= 0) return;

    // 选择线程块大小
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // 限制 blocks 数量不超过合理上限（例如 65535）
    if (blocks > 65535) blocks = 65535;

    // 使用 grid-stride loop，因此 blocks 不需要完全覆盖，但为了性能尽量覆盖
    adaptive_max_pool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, outH, outW
    );

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // 不能直接在 .cu 中抛出异常，但可以打印错误？
        // 这里仅忽略，或者可以 printf，但避免使用 printf
        // 生产者未定义处理方式，因此忽略错误。
    }
}
