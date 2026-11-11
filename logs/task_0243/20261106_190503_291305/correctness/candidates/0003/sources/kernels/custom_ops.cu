#include <cuda_runtime.h>

// 3D average pooling kernel for kernel_size=(2,2,2), stride=(2,2,2), padding=0
// Input tensor shape: [N, C, D, H, W]
// Output tensor shape: [N, C, D_out, H_out, W_out]
// Output is assumed contiguous; input may have arbitrary strides.
__global__ void avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int in_stride_n, int in_stride_c, int in_stride_d, int in_stride_h, int in_stride_w)
{
    int total_outputs = N * C * D_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int linear = idx; linear < total_outputs; linear += stride) {
        // Decompose linear index into n, c, od, oh, ow
        int ow = linear % W_out;
        int tmp = linear / W_out;
        int oh = tmp % H_out;
        tmp /= H_out;
        int od = tmp % D_out;
        tmp /= D_out;
        int c = tmp % C;
        int n = tmp / C;

        // Corresponding input start indices (stride 2, no padding)
        int in_d = od * 2;
        int in_h = oh * 2;
        int in_w = ow * 2;

        // Accumulate over the 2x2x2 window
        float sum = 0.0f;
        sum += input[n * in_stride_n + c * in_stride_c + in_d * in_stride_d + in_h * in_stride_h + in_w * in_stride_w];
        sum += input[n * in_stride_n + c * in_stride_c + in_d * in_stride_d + in_h * in_stride_h + (in_w + 1) * in_stride_w];
        sum += input[n * in_stride_n + c * in_stride_c + in_d * in_stride_d + (in_h + 1) * in_stride_h + in_w * in_stride_w];
        sum += input[n * in_stride_n + c * in_stride_c + in_d * in_stride_d + (in_h + 1) * in_stride_h + (in_w + 1) * in_stride_w];
        sum += input[n * in_stride_n + c * in_stride_c + (in_d + 1) * in_stride_d + in_h * in_stride_h + in_w * in_stride_w];
        sum += input[n * in_stride_n + c * in_stride_c + (in_d + 1) * in_stride_d + in_h * in_stride_h + (in_w + 1) * in_stride_w];
        sum += input[n * in_stride_n + c * in_stride_c + (in_d + 1) * in_stride_d + (in_h + 1) * in_stride_h + in_w * in_stride_w];
        sum += input[n * in_stride_n + c * in_stride_c + (in_d + 1) * in_stride_d + (in_h + 1) * in_stride_h + (in_w + 1) * in_stride_w];

        output[linear] = sum / 8.0f;
    }
}

// 3D convolution kernel: direct implementation, supports stride and padding
// Input: x [N, C_in, D, H, W]
// Weight: w [K, C_in, kD, kH, kW]
// Bias: optional, [K]
// Output: y [N, K, D_out, H_out, W_out]
// Stride and padding are per-dimension (3 values)
// This kernel assumes input and output are contiguous, weight is contiguous.
__global__ void conv3d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int D, int H, int W,
    int K, int kD, int kH, int kW,
    int D_out, int H_out, int W_out,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    bool has_bias)
{
    int total_outputs = N * K * D_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int linear = idx; linear < total_outputs; linear += stride) {
        // Decompose linear index
        int ow = linear % W_out;
        int tmp = linear / W_out;
        int oh = tmp % H_out;
        tmp /= H_out;
        int od = tmp % D_out;
        tmp /= D_out;
        int k = tmp % K;
        int n = tmp / K;

        float acc = 0.0f;
        for (int c = 0; c < C_in; ++c) {
            for (int kd = 0; kd < kD; ++kd) {
                int in_d = od * stride_d - pad_d + kd;
                if (in_d < 0 || in_d >= D) continue;
                for (int kh = 0; kh < kH; ++kh) {
                    int in_h = oh * stride_h - pad_h + kh;
                    if (in_h < 0 || in_h >= H) continue;
                    for (int kw = 0; kw < kW; ++kw) {
                        int in_w = ow * stride_w - pad_w + kw;
                        if (in_w < 0 || in_w >= W) continue;
                        // Input index
                        int input_idx = ((n * C_in + c) * D + in_d) * H * W + in_h * W + in_w;
                        // Weight index: k, c, kd, kh, kw
                        int weight_idx = (((k * C_in + c) * kD + kd) * kH + kh) * kW + kw;
                        acc += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
        if (has_bias) {
            acc += bias[k];
        }
        output[linear] = acc;
    }
}

extern "C" void avg_pool3d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int in_stride_n, int in_stride_c, int in_stride_d, int in_stride_h, int in_stride_w,
    cudaStream_t stream)
{
    int total_outputs = N * C * D_out * H_out * W_out;
    if (total_outputs <= 0) return;

    const int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    const int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;

    avg_pool3d_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, D, H, W,
        D_out, H_out, W_out,
        in_stride_n, in_stride_c, in_stride_d, in_stride_h, in_stride_w);
}

extern "C" void conv3d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int D, int H, int W,
    int K, int kD, int kH, int kW,
    int D_out, int H_out, int W_out,
    int stride_d, int stride_h, int stride_w,
    int pad_d, int pad_h, int pad_w,
    bool has_bias,
    cudaStream_t stream)
{
    int total_outputs = N * K * D_out * H_out * W_out;
    if (total_outputs <= 0) return;

    const int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    const int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;

    conv3d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, D, H, W,
        K, kD, kH, kW,
        D_out, H_out, W_out,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        has_bias);
}
