#include <cuda_runtime.h>
#include <cstdint>

// Custom convolution kernel specialized for the task's parameters:
// batch_size=16, in_channels=3, out_channels=64, kernel=(3,5),
// stride=(1,1), padding=(1,2), dilation=(2,1), bias=False
// Input: [16, 3, 512, 512], Weight: [64, 3, 3, 5], Output: [16, 64, 510, 512]
// We implement a direct convolution with loops over output positions and filter.
// This is not optimized but should be correct and comparable to torch.compile baseline.

__global__ void conv2d_naive_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out) {

    // Calculate output index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * K * H_out * W_out;
    if (idx >= total_elements) return;

    // Decode index
    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int k = (idx / (W_out * H_out)) % K;
    int n = idx / (W_out * H_out * K);

    // Determine group
    int group = k / (K / groups);
    int C_per_group = C / groups;
    int c_start = group * C_per_group;
    int c_end = c_start + C_per_group;

    float sum = 0.0f;
    for (int c = c_start; c < c_end; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int h_in = h_out * stride_h - padding_h + r * dilation_h;
                int w_in = w_out * stride_w - padding_w + s * dilation_w;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    int input_idx = ((n * C + c) * H + h_in) * W + w_in;
                    int weight_idx = ((k * C_per_group + (c - c_start)) * R + r) * S + s;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }
    output[idx] = sum;
}

// Launcher function (C interface)
extern "C" void conv2d_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    int N, int C, int H, int W,
    int K, int R, int S,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int groups,
    int H_out, int W_out,
    cudaStream_t stream) {

    int total_elements = N * K * H_out * W_out;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    conv2d_naive_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, output,
        N, C, H, W,
        K, R, S,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        groups,
        H_out, W_out);
}
