// CUDA kernels for MaxPool2d and ReLU
#include <cuda_runtime.h>
#include <cstdint>

// Forward declarations for kernels used in launchers
__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int total
);

__global__ void relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
);

// MaxPool2d launcher
extern "C" void maxpool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    cudaStream_t stream
) {
    int H_out = (H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Using grid-stride loop kernel
    maxpool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, H_out, W_out,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, total);
}

// ReLU launcher
extern "C" void relu_forward_launcher(
    const float* input,
    float* output,
    int n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

// Kernel implementations
__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int w_out = i % W_out;
        int h_out = (i / W_out) % H_out;
        int c = (i / (W_out * H_out)) % C;
        int n = i / (W_out * H_out * C);
        float max_val = -1e30f;
        int h_start = h_out * stride_h - padding_h;
        int w_start = w_out * stride_w - padding_w;
        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_in = h_start + kh * dilation_h;
            if (h_in < 0 || h_in >= H) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int w_in = w_start + kw * dilation_w;
                if (w_in < 0 || w_in >= W) continue;
                float val = input[((n * C + c) * H + h_in) * W + w_in];
                if (val > max_val) max_val = val;
            }
        }
        output[i] = max_val;
    }
}

__global__ void relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        output[i] = fmaxf(input[i], 0.0f);
    }
}
