// 移除 cuDNN 依赖，仅包含 CUDA kernel 和 launcher
#include <cuda_runtime.h>
#include <cstdint>

// SiLU kernel (保持不变)
__global__ void silu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t n
) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += stride) {
        float x = input[i];
        float sig;
        if (x >= 0.0f) {
            sig = 1.0f / (1.0f + __expf(-x));
        } else {
            float ex = __expf(x);
            sig = ex / (1.0f + ex);
        }
        output[i] = x * sig;
    }
}

extern "C" cudaError_t custom_silu_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream
) {
    if (numel <= 0) return cudaSuccess;
    const int threads = 256;
    int64_t blocks64 = (numel + threads - 1) / threads;
    if (blocks64 > 2147483647LL) blocks64 = 2147483647LL;
    int blocks = static_cast<int>(blocks64);
    silu_kernel<<<blocks, threads, 0, stream>>>(output, input, numel);
    return cudaGetLastError();
}

// --- ConvTranspose1d kernel (自定义实现) ---
// 输入布局: x[N][C_in][L_in], weight[C_in][C_out][K]
// 输出: y[N][C_out][L_out], L_out = (L_in-1)*stride - 2*padding + K + output_padding
// 每个线程计算一个输出元素
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // 可为 nullptr
    float* __restrict__ y,
    int N, int C_in, int C_out, int L_in, int L_out,
    int K, int stride, int padding, int output_padding,
    bool has_bias
) {
    int64_t total = (int64_t)N * C_out * L_out;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int n = idx / (C_out * L_out);
    int rem = idx % (C_out * L_out);
    int c_out = rem / L_out;
    int l_out = rem % L_out;

    // 计算核心输出长度（不含 output_padding）
    int L_core = (L_in - 1) * stride - 2 * padding + K;
    // 若 l_out >= L_core，则只可能是 output_padding 区域，应保持 0（bias 会加吗？参考 PyTorch，output_padding 增加的空间在卷积结果之后，bias 会加到所有输出元素，包括 output_padding 的部分？实际上 PyTorch ConvTranspose 的 output_padding 是在核心输出后面填充零，bias 会加到整个输出张量，包括填充部分吗？需要检查。根据 PyTorch 实现，bias 是加到卷积结果上，然后 output_padding 是在最后添加的零列，bias 应该也会加到那些位置？测试：通常 output_padding 增加的位置 bias 也会被加上，因为 bias 是 per-channel，加到所有输出位置。但我们需要与参考一致，步骤中 bias 在 conv_transpose 内部应用，output_padding 也是 conv 的一部分，所以 bias 应该加到包含 output_padding 的所有输出元素。我们按此实现。
    float acc = 0.0f;

    // 对每个输入通道和 kernel 位置累加
    for (int c_in = 0; c_in < C_in; ++c_in) {
        // 找到对输出位置 l_out 有贡献的输入位置和 kernel 索引
        // 转置卷积公式：y[n, c_out, l_out] = sum_{c_in, k} x[n, c_in, l_in] * weight[c_in, c_out, k]
        // 其中 l_out = l_in * stride - padding + k
        // 所以 l_in = (l_out + padding - k) / stride，且必须为整数且 0 <= l_in < L_in
        for (int k = 0; k < K; ++k) {
            int pos = l_out + padding - k;
            if (pos < 0) continue;
            if (pos % stride != 0) continue;
            int l_in = pos / stride;
            if (l_in < 0 || l_in >= L_in) continue;
            // 访问输入和权重
            int x_idx = ((n * C_in + c_in) * L_in + l_in);
            int w_idx = (c_in * C_out + c_out) * K + k;
            acc += x[x_idx] * weight[w_idx];
        }
    }

    if (has_bias) {
        acc += bias[c_out];
    }
    y[idx] = acc;
}

extern "C" cudaError_t custom_conv_transpose1d_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int64_t N,
    int64_t C_in,
    int64_t C_out,
    int64_t L_in,
    int64_t L_out,
    int64_t K,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    bool has_bias,
    cudaStream_t stream
) {
    int64_t total = N * C_out * L_out;
    if (total <= 0) return cudaSuccess;
    const int threads = 256;
    int64_t blocks64 = (total + threads - 1) / threads;
    if (blocks64 > 2147483647LL) blocks64 = 2147483647LL;
    int blocks = static_cast<int>(blocks64);
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, bias, y,
        (int)N, (int)C_in, (int)C_out, (int)L_in, (int)L_out,
        (int)K, (int)stride, (int)padding, (int)output_padding, has_bias);
    return cudaGetLastError();
}
