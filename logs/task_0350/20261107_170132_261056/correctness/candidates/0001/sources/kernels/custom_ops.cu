#include <cuda_runtime.h>

// ReLU forward: y = max(x, 0) elementwise
extern "C" __global__ void relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long numel) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

// MaxPool1d forward: kernel_size=2, stride=2, padding=0, dilation=1
// input shape: [batch, channels, length_in]
// output shape: [batch, channels, length_out]
extern "C" __global__ void max_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int length_in,
    int length_out,
    int kernel_size,
    int stride,
    int padding) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= batch_size * channels * length_out) return;

    int out_w = out_idx % length_out;
    int tmp = out_idx / length_out;
    int c = tmp % channels;
    int b = tmp / channels;

    int in_start = out_w * stride - padding;
    int in_end = in_start + kernel_size;
    if (in_start < 0) in_start = 0;
    if (in_end > length_in) in_end = length_in;

    float max_val = -3.402823466e+38f; // -FLT_MAX
    for (int in_w = in_start; in_w < in_end; ++in_w) {
        float val = input[((long long)b * channels + c) * length_in + in_w];
        if (val > max_val) max_val = val;
    }
    output[out_idx] = max_val;
}

// Launcher for ReLU
extern "C" void relu_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream) {
    int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

// Launcher for MaxPool1d
extern "C" void max_pool1d_forward_launcher(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int length_in,
    int length_out,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream) {
    int total_out = batch_size * channels * length_out;
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    max_pool1d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch_size, channels, length_in, length_out,
        kernel_size, stride, padding);
}
