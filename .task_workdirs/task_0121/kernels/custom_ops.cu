#include <cuda_runtime.h>
#include <stdint.h>

// Kernel: pad1d forward
// x: [B, C, H, W] float32 contiguous
// out: [B, C, H, W + pad_left + pad_right] float32 contiguous
__global__ void pad1d_kernel(const float* __restrict__ x,
                             float* __restrict__ out,
                             int B, int C, int H, int W,
                             int pad_left, int pad_right,
                             int out_W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * out_W;
    if (idx >= total) return;

    // 解算输出坐标
    int w_out = idx % out_W;
    int tmp = idx / out_W;
    int h = tmp % H;
    tmp /= H;
    int c = tmp % C;
    int b = tmp / C;

    int w_in = w_out - pad_left;
    if (w_in >= 0 && w_in < W) {
        int in_idx = ((b * C + c) * H + h) * W + w_in;
        out[idx] = x[in_idx];
    } else {
        out[idx] = 0.0f;
    }
}

// Launcher: pad1d
// x: input pointer, out: output pointer (preallocated by binding)
// B, C, H, W: input dimensions
// pad_left, pad_right: padding on left and right
// stream: CUDA stream
extern "C" void custom_pad1d_forward_launcher(const float* x,
                                              float* out,
                                              int B, int C, int H, int W,
                                              int pad_left, int pad_right,
                                              cudaStream_t stream) {
    int out_W = W + pad_left + pad_right;
    int total = B * C * H * out_W;
    if (total == 0) return;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    pad1d_kernel<<<blocks, threads, 0, stream>>>(x, out, B, C, H, W,
                                                 pad_left, pad_right, out_W);
}

// Kernel: add_bias forward
// x: [B, C, H, W] float32 contiguous
// bias: [C] float32 (from bias shape [C,1,1])
// out: same shape as x, out[i] = x[i] + bias[c]
__global__ void add_bias_kernel(const float* __restrict__ x,
                                const float* __restrict__ bias,
                                float* __restrict__ out,
                                int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * H * W;
    if (idx >= total) return;

    int spatial = H * W;
    int c = (idx / spatial) % C;
    out[idx] = x[idx] + bias[c];
}

// Launcher: add_bias
// x: input pointer, bias: bias pointer (C elements), out: output pointer (same shape as x)
// B, C, H, W: dimensions
// stream: CUDA stream
extern "C" void custom_add_bias_forward_launcher(const float* x,
                                                 const float* bias,
                                                 float* out,
                                                 int B, int C, int H, int W,
                                                 cudaStream_t stream) {
    int total = B * C * H * W;
    if (total == 0) return;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(x, bias, out, B, C, H, W);
}

// Declaration of conv_transpose2d launcher for binding (to be implemented in C++ using cuDNN)
extern "C" void custom_conv_transpose2d_forward_launcher(
    const float* x, const float* weight, const float* bias,
    float* out,
    int B, int C_in, int H_in, int W_in,
    int C_out, int kH, int kW,
    int stride, int output_padding,
    cudaStream_t stream);
