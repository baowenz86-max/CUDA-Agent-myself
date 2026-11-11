#include <cuda_runtime.h>

// channel_mean_forward: 沿 dim=1 (通道维) 求均值, keepdim=True
// 输入: x [N, C, H, W], 输出: out [N, 1, H, W]
// 假设 C=3, 但代码支持任意 C
__global__ void channel_mean_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = N * H * W;
    for (int i = idx; i < total; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w = hw % W;
        const float* base = input + n * C * H * W + h * W + w;
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            sum += base[c * H * W];
        }
        output[i] = sum / C;
    }
}

// max_pool2d_forward: 标准 2D 最大池化
// 输入: x [N, C, H, W], 输出: out [N, C, outH, outW]
// 参数: kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w
__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = N * C * outH * outW;
    for (int i = idx; i < total; i += stride) {
        int ow = i % outW;
        int oh = (i / outW) % outH;
        int oc = (i / (outW * outH)) % C;
        int n = i / (outW * outH * C);

        int h_start = oh * stride_h - padding_h;
        int w_start = ow * stride_w - padding_w;
        int h_end = h_start + kernel_h;
        int w_end = w_start + kernel_w;

        float max_val = -3.402823466e+38F; // -FLT_MAX
        bool found = false;

        for (int kh = 0; kh < kernel_h; ++kh) {
            int h = h_start + kh;
            if (h < 0 || h >= H) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int w = w_start + kw;
                if (w < 0 || w >= W) continue;
                float val = input[n * C * H * W + oc * H * W + h * W + w];
                if (!found || val > max_val) {
                    max_val = val;
                    found = true;
                }
            }
        }
        // 如果整个 kernel 窗口都在 padding 外，输出 -inf? 参考 PyTorch 行为：
        // 实际上 MaxPool2d 不会出现完全 padding 的情况，因为输出尺寸定义保证了至少一个有效元素。
        // 但为安全起见，若未找到任何有效值，输出 -inf。
        output[i] = found ? max_val : -3.402823466e+38F;
    }
}

// launcher for channel_mean_forward
extern "C" void channel_mean_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    cudaStream_t stream) {
    int total = N * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // 限制 blocks 不超过 GPU 上限，使用 grid-stride 循环处理所有元素
    if (blocks > 65535) blocks = 65535;
    channel_mean_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W);
}

// launcher for max_pool2d_forward
extern "C" void max_pool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    cudaStream_t stream) {
    int outH = (H + 2 * padding_h - kernel_h) / stride_h + 1;
    int outW = (W + 2 * padding_w - kernel_w) / stride_w + 1;
    int total = N * C * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    max_pool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, outH, outW,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
}
