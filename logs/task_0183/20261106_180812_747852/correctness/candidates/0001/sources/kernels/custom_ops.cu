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

// Placeholder for conv2d using cuDNN is intentionally omitted here.
// The binding for conv2d will be implemented in a separate .cpp file using cuDNN API.
// This .cu file only contains non-convolution CUDA kernels.
