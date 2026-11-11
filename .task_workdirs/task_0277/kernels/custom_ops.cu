#include <cuda_runtime.h>

// SELU activation: scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
// alpha = 1.6732632423543772, scale = 1.0507009873554805
__global__ void selu_kernel(float* out, const float* in, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    const float alpha = 1.6732632423543772f;
    const float scale = 1.0507009873554805f;

    for (int64_t i = idx; i < numel; i += stride) {
        float x = in[i];
        float val = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
        out[i] = scale * val;
    }
}

// Conv2d forward: input (N, C, H, W), weight (K, C_per_group, kH, kW), output (N, K, H_out, W_out)
// Supports stride, padding, dilation, groups. Assumes float32, NCHW layout, contiguous tensors.
__global__ void conv2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C, int H, int W,
    int K, int C_per_group, int kH, int kW,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out
) {
    int total = N * K * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int k = tmp % K;
    int n = tmp / K;

    int group = k / (K / groups);
    int c_start = group * C_per_group;
    int c_end = c_start + C_per_group;

    float accum = bias != nullptr ? bias[k] : 0.0f;

    for (int c = c_start; c < c_end; ++c) {
        for (int kh = 0; kh < kH; ++kh) {
            int in_h = h_out * stride_h - padding_h + kh * dilation_h;
            if (in_h < 0 || in_h >= H) continue;
            for (int kw = 0; kw < kW; ++kw) {
                int in_w = w_out * stride_w - padding_w + kw * dilation_w;
                if (in_w < 0 || in_w >= W) continue;
                float input_val = input[((n * C + c) * H + in_h) * W + in_w];
                float weight_val = weight[((k * C_per_group + (c - c_start)) * kH + kh) * kW + kw];
                accum += input_val * weight_val;
            }
        }
    }

    output[((n * K + k) * H_out + h_out) * W_out + w_out] = accum;
}

extern "C" void selu_forward_launcher(
    float* out,
    const float* in,
    int64_t numel,
    cudaStream_t stream
) {
    const int block_size = 256;
    int64_t max_grid_x = (numel + block_size - 1) / block_size;
    if (max_grid_x < 1) max_grid_x = 1;
    if (max_grid_x > 2147483647LL) max_grid_x = 2147483647LL;
    int grid_x = (int)max_grid_x;
    selu_kernel<<<grid_x, block_size, 0, stream>>>(out, in, numel);
}

extern "C" void conv2d_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C, int H, int W,
    int K, int C_per_group, int kH, int kW,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out,
    cudaStream_t stream
) {
    int total = N * K * H_out * W_out;
    if (total <= 0) return;
    const int block_size = 256;
    int grid_x = (total + block_size - 1) / block_size;
    if (grid_x > 2147483647) grid_x = 2147483647;
    conv2d_kernel<<<grid_x, block_size, 0, stream>>>(
        output, input, weight, bias,
        N, C, H, W,
        K, C_per_group, kH, kW,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups,
        H_out, W_out
    );
}
