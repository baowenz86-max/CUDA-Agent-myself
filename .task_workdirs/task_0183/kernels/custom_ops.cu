// CUDA kernels for custom ops used in model_new.py
// This file contains only CUDA kernels and launchers without torch dependencies.
// All launchers are extern "C" for binding layer.

#include <cuda_runtime.h>

// Helper for reflection padding index calculation
__device__ __forceinline__ int reflect_index(int idx, int dim) {
    if (idx < 0) idx = -idx;
    if (idx >= dim) idx = 2 * dim - idx - 2;
    return idx;
}

// Kernel for ReflectionPad2d (padding=1 on last two dimensions)
// Input: [batch, channels, height, width], supports arbitrary strides
// Output: [batch, channels, height+2*pad, width+2*pad]
__global__ void reflection_pad2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int pad,
    long long in_stride_b,
    long long in_stride_c,
    long long in_stride_h,
    long long in_stride_w,
    long long out_stride_b,
    long long out_stride_c,
    long long out_stride_h,
    long long out_stride_w)
{
    int out_h = in_h + 2 * pad;
    int out_w = in_w + 2 * pad;
    long long total = (long long)batch * channels * out_h * out_w;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        int w = idx % out_w;
        long long tmp = idx / out_w;
        int h = tmp % out_h;
        tmp /= out_h;
        int c = tmp % channels;
        int b = tmp / channels;

        int src_h = reflect_index(h - pad, in_h);
        int src_w = reflect_index(w - pad, in_w);

        long long src_idx = (long long)b * in_stride_b + (long long)c * in_stride_c
                          + (long long)src_h * in_stride_h + (long long)src_w * in_stride_w;
        long long dst_idx = (long long)b * out_stride_b + (long long)c * out_stride_c
                          + (long long)h * out_stride_h + (long long)w * out_stride_w;

        output[dst_idx] = input[src_idx];
    }
}

// Launcher for ReflectionPad2d
extern "C" void reflection_pad2d_launcher(
    float* output,
    const float* input,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int pad,
    long long in_stride_b,
    long long in_stride_c,
    long long in_stride_h,
    long long in_stride_w,
    long long out_stride_b,
    long long out_stride_c,
    long long out_stride_h,
    long long out_stride_w,
    cudaStream_t stream)
{
    int out_h = in_h + 2 * pad;
    int out_w = in_w + 2 * pad;
    long long total = (long long)batch * channels * out_h * out_w;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid-stride loop handles rest
    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        output, input, batch, channels, in_h, in_w, pad,
        in_stride_b, in_stride_c, in_stride_h, in_stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w);
}

// Direct convolution kernel (no cuDNN, no cuBLAS)
// Assumes input, weight, output are contiguous in NCHW format
// Input: [N, C, H, W]
// Weight: [K, C, R, S]
// Output: [N, K, P, Q] where P=H+2*pad_h-R+1, Q=W+2*pad_w-S+1
__global__ void conv2d_direct_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int P, int Q)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= N * K * P * Q) return;

    int q = out_idx % Q;
    int tmp = out_idx / Q;
    int p = tmp % P;
    tmp /= P;
    int k = tmp % K;
    int n = tmp / K;

    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            int h_in = p * stride_h - pad_h + r;
            if (h_in < 0 || h_in >= H) continue;
            for (int s = 0; s < S; ++s) {
                int w_in = q * stride_w - pad_w + s;
                if (w_in < 0 || w_in >= W) continue;

                int input_idx = ((n * C + c) * H + h_in) * W + w_in;
                int weight_idx = ((k * C + c) * R + r) * S + s;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    if (bias != nullptr) {
        sum += bias[k];
    }
    output[out_idx] = sum;
}

// Launcher for conv2d direct (contiguous input/weight/output)
extern "C" void conv2d_direct_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int P, int Q,
    cudaStream_t stream)
{
    int total = N * K * P * Q;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv2d_direct_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C, H, W, K, R, S,
        pad_h, pad_w, stride_h, stride_w, P, Q);
}
