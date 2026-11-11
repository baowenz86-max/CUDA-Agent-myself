#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include <cfloat>

// 自定义算子：绝对值与最大池化

// 逐元素绝对值 kernel
__global__ void abs_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t total_elements) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        output[idx] = fabsf(input[idx]);
    }
}

// 最大池化 kernel: 每个输出元素一个线程
__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation) {
    
    int64_t out_h_total = (H + 2 * pad_h - dilation * (kernel_h - 1) - 1) / stride + 1;
    int64_t out_w_total = (W + 2 * pad_w - dilation * (kernel_w - 1) - 1) / stride + 1;
    int64_t total_outputs = N * C * out_h_total * out_w_total;
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) return;
    
    // 从线性索引恢复 n,c,oh,ow
    int64_t ow = idx % out_w_total;
    int64_t oh = (idx / out_w_total) % out_h_total;
    int64_t c = (idx / (out_w_total * out_h_total)) % C;
    int64_t n = idx / (out_w_total * out_h_total * C);
    
    float max_val = -FLT_MAX;
    bool found = false;
    
    for (int64_t kh = 0; kh < kernel_h; ++kh) {
        int64_t ih = oh * stride - pad_h + kh * dilation;
        if (ih < 0 || ih >= H) continue;
        for (int64_t kw = 0; kw < kernel_w; ++kw) {
            int64_t iw = ow * stride - pad_w + kw * dilation;
            if (iw < 0 || iw >= W) continue;
            float val = input[((n * C + c) * H + ih) * W + iw];
            if (!found || val > max_val) {
                max_val = val;
                found = true;
            }
        }
    }
    
    // 如果没有有效元素，用 -inf (但我们的参数下不会发生)
    output[idx] = found ? max_val : -FLT_MAX;
}

// 绝对值 launcher
extern "C" void custom_abs_launcher(
    float* output,
    const float* input,
    int64_t total_elements,
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    abs_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements);
}

// 最大池化 launcher（支持非对称 kernel 和 padding）
extern "C" void custom_max_pool2d_launcher(
    float* output,
    const float* input,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride,
    int64_t pad_h,
    int64_t pad_w,
    int64_t dilation,
    cudaStream_t stream) {
    
    int64_t out_h = (H + 2 * pad_h - dilation * (kernel_h - 1) - 1) / stride + 1;
    int64_t out_w = (W + 2 * pad_w - dilation * (kernel_w - 1) - 1) / stride + 1;
    int64_t total_outputs = N * C * out_h * out_w;
    
    int threads = 256;
    int blocks = (int)((total_outputs + threads - 1) / threads);
    max_pool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, kernel_h, kernel_w, stride, pad_h, pad_w, dilation);
}
