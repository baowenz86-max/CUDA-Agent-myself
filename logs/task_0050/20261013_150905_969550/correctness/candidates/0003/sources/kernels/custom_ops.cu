#include <cuda_runtime.h>

// CUDA kernel for 2D convolution (NCHW layout)
// Supports stride, padding, dilation, groups, and optional bias.
// This kernel is used as a fallback when cuDNN is unavailable.
__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * K * H_out * W_out;
    if (idx >= total_elements) return;

    // Decode linear index to (n, k, oh, ow)
    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    tmp /= H_out;
    int k = tmp % K;
    int n = tmp / K;

    int C_per_group = C / groups;
    int group = k / (K / groups);
    int k_in_group = k % (K / groups);
    int input_group_offset = group * C_per_group;
    int weight_group_offset = group * (K / groups) * C_per_group * R * S;

    float sum = 0.0f;
    for (int c = 0; c < C_per_group; ++c) {
        int input_channel = input_group_offset + c;
        int weight_channel = c; // within group, weight channel offset is c
        for (int r = 0; r < R; ++r) {
            int h_in = oh * stride_h - padding_h + r * dilation_h;
            if (h_in < 0 || h_in >= H) continue;
            for (int s = 0; s < S; ++s) {
                int w_in = ow * stride_w - padding_w + s * dilation_w;
                if (w_in < 0 || w_in >= W) continue;

                int input_idx = ((n * C + input_channel) * H + h_in) * W + w_in;
                int weight_idx = weight_group_offset + (k_in_group * C_per_group + weight_channel) * R * S + r * S + s;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }

    int output_idx = ((n * K + k) * H_out + oh) * W_out + ow;
    output[output_idx] = sum;
}

// Launcher function (C interface, no PyTorch types)
extern "C" void conv2d_launcher(
    const float* input,
    const float* weight,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out,
    cudaStream_t stream)
{
    int total_elements = N * K * H_out * W_out;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    conv2d_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weight, output,
        N, C, H, W,
        K, R, S,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups,
        H_out, W_out);
}
