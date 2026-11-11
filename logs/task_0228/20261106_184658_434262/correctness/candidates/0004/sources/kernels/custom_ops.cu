#include <cuda_runtime.h>

// 辅助函数：获取全局线程索引
typedef unsigned int uint;

// triu_forward kernel：对每个 batch 和 channel 的最后两维执行上三角掩码
// 输入形状: [N, C, H, W]，连续存储
// 输出与输入形状相同
__global__ void triu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int N, int C, int H, int W,
                            uint total_elements) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;

    for (uint i = idx; i < total_elements; i += stride) {
        // 计算 n, c, h, w
        int w = i % W;
        int tmp = i / W;
        int h = tmp % H;
        tmp = tmp / H;
        int c = tmp % C;
        int n = tmp / C;

        // 上三角：保留 h <= w 的元素，否则置零
        if (h <= w) {
            output[i] = input[i];
        } else {
            output[i] = 0.0f;
        }
    }
}

// batch_norm_eval_forward kernel：每个元素独立归一化
// 输入形状: [N, C, H, W]，参数形状: [C]
__global__ void batch_norm_eval_kernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       const float* __restrict__ running_mean,
                                       const float* __restrict__ running_var,
                                       const float* __restrict__ weight,
                                       const float* __restrict__ bias,
                                       float eps,
                                       int N, int C, int H, int W,
                                       uint total_elements) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;

    for (uint i = idx; i < total_elements; i += stride) {
        // 计算通道索引 c
        int w = i % W;
        int tmp = i / W;
        int h = tmp % H;
        tmp = tmp / H;
        int c = tmp % C;

        float mean = running_mean[c];
        float var = running_var[c];
        float gamma = weight ? weight[c] : 1.0f;  // 如果 weight 为 null，则视为 1
        float beta = bias ? bias[c] : 0.0f;      // 如果 bias 为 null，则视为 0

        float inv_std = rsqrtf(var + eps);
        y[i] = (x[i] - mean) * inv_std * gamma + beta;
    }
}

// launcher: triu_forward
// 参数：输入指针，输出指针，N, C, H, W, stream
extern "C" void triu_forward_launcher(const float* input,
                                     float* output,
                                     int N, int C, int H, int W,
                                     cudaStream_t stream) {
    uint total = (uint)N * C * H * W;
    if (total == 0) return;

    // 选择 block 大小和 grid 大小，使用 grid-stride 循环
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // 限制 grid 大小，配合 grid-stride

    triu_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W, total);
}

// launcher: batch_norm_eval_forward
// 参数：输入指针，输出指针，running_mean, running_var, weight, bias, eps, N, C, H, W, stream
extern "C" void batch_norm_eval_forward_launcher(const float* input,
                                                 float* output,
                                                 const float* running_mean,
                                                 const float* running_var,
                                                 const float* weight,
                                                 const float* bias,
                                                 float eps,
                                                 int N, int C, int H, int W,
                                                 cudaStream_t stream) {
    uint total = (uint)N * C * H * W;
    if (total == 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    batch_norm_eval_kernel<<<blocks, threads, 0, stream>>>(
        input, output, running_mean, running_var, weight, bias, eps,
        N, C, H, W, total);
}
