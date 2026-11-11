#include <cuda_runtime.h>

// 直接卷积的 CUDA kernel，支持分组卷积、bias 和 ReLU
// 每个线程计算一个输出元素
__global__ void conv2d_relu_kernel(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t K, int64_t R, int64_t S,
    int64_t stride, int64_t padding, int64_t dilation,
    int64_t groups) {

    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = N * K * ((H + 2 * padding - dilation * (R - 1) - 1) / stride + 1) * ((W + 2 * padding - dilation * (S - 1) - 1) / stride + 1);
    if (idx >= total) return;

    // 从线性索引还原坐标
    int64_t out_w = (W + 2 * padding - dilation * (S - 1) - 1) / stride + 1;
    int64_t out_h = (H + 2 * padding - dilation * (R - 1) - 1) / stride + 1;
    int64_t out_size = out_h * out_w;
    int64_t n = idx / (K * out_size);
    int64_t rem = idx % (K * out_size);
    int64_t k = rem / out_size;
    int64_t spatial = rem % out_size;
    int64_t oh = spatial / out_w;
    int64_t ow = spatial % out_w;

    int64_t C_per_group = C / groups;
    int64_t group = k / (K / groups); // 注意 K 必须能被 groups 整除
    int64_t k_in_group = k % (K / groups);

    float sum = 0.0f;
    for (int64_t c = 0; c < C_per_group; ++c) {
        int64_t channel = group * C_per_group + c;
        for (int64_t r = 0; r < R; ++r) {
            for (int64_t s = 0; s < S; ++s) {
                int64_t h_in = oh * stride - padding + r * dilation;
                int64_t w_in = ow * stride - padding + s * dilation;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    float input_val = input[((n * C + channel) * H + h_in) * W + w_in];
                    float weight_val = weight[(((k)*C_per_group + c) * R + r) * S + s];
                    sum += input_val * weight_val;
                }
            }
        }
    }

    if (bias != nullptr) {
        sum += bias[k];
    }

    // ReLU
    output[idx] = fmaxf(sum, 0.0f);
}

// launcher
// 该函数导出为 C ABI，供 binding.cpp 调用
extern "C" void conv2d_relu_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t K, int64_t R, int64_t S,
    int64_t stride, int64_t padding, int64_t dilation,
    int64_t groups,
    cudaStream_t stream) {

    int64_t out_h = (H + 2 * padding - dilation * (R - 1) - 1) / stride + 1;
    int64_t out_w = (W + 2 * padding - dilation * (S - 1) - 1) / stride + 1;
    int64_t total = N * K * out_h * out_w;

    int threads = 256;
    long long blocks_ll = (total + threads - 1) / threads;
    if (blocks_ll > 0) {
        int blocks = (int)blocks_ll;
        conv2d_relu_kernel<<<blocks, threads, 0, stream>>>(
            output, input, weight, bias,
            N, C, H, W,
            K, R, S,
            stride, padding, dilation,
            groups
        );
    }
}