#include <cuda_runtime.h>

// CUDA kernel for torch.triu on the last two dimensions.
// Input shape (N, C, H, W) or any tensor with at least 2 dims.
// Applies upper triangular mask to last two dims (H, W).
// Handles arbitrary strides via multi-dimensional index computation.
__global__ void triu_kernel(float* data,
                            int total_elements,
                            int H, int W, int C,
                            long long stride_batch,
                            long long stride_channel,
                            long long stride_h,
                            long long stride_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_elements; i += stride) {
        // Reconstruct multi-dimensional indices
        int w = i % W;
        int h = (i / W) % H;
        int c = (i / (H * W)) % C;
        int n = i / (H * W * C);

        long long offset = n * stride_batch + c * stride_channel + h * stride_h + w * stride_w;

        // For triu, keep if h <= w (diagonal and above)
        if (h > w) {
            data[offset] = 0.0f;
        }
    }
}

// Launcher for triu_forward: modifies input in-place, returns same tensor.
// Assumes output pointer is valid and same shape as input.
extern "C" void triu_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    long long stride_batch, long long stride_channel, long long stride_h, long long stride_w,
    cudaStream_t stream)
{
    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    triu_kernel<<<blocks, threads, 0, stream>>>(
        output, total, H, W, C,
        stride_batch, stride_channel, stride_h, stride_w);
}

// Bias add kernel for x + bias where x shape (N,C,H,W) and bias shape (C,1,1) broadcast.
// Handles arbitrary strides.
__global__ void bias_add_kernel(
    const float* x,
    const float* bias,
    float* output,
    int total_elements,
    int C, int H, int W,
    long long stride_batch, long long stride_channel, long long stride_h, long long stride_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_elements; i += stride) {
        int w = i % W;
        int h = (i / W) % H;
        int c = (i / (H * W)) % C;
        int n = i / (H * W * C);  // batch index n
        long long offset = n * stride_batch + c * stride_channel + h * stride_h + w * stride_w;
        output[offset] = x[offset] + bias[c];
    }
}

// Launcher for bias_add_forward: output = x + bias.
extern "C" void bias_add_forward_launcher(
    const float* x,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    long long stride_batch, long long stride_channel, long long stride_h, long long stride_w,
    cudaStream_t stream)
{
    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    bias_add_kernel<<<blocks, threads, 0, stream>>>(
        x, bias, output, total, C, H, W,
        stride_batch, stride_channel, stride_h, stride_w);
}

// ---- Convolution kernel (direct convolution) ----
// Implements 2D convolution with stride 1, padding 0, dilation 1, groups 1.
// Input: N x C x H x W, Weight: K x C x kH x kW, Output: N x K x outH x outW.
// All tensors are assumed contiguous (as PyTorch tensors are for this task).
__global__ void conv2d_direct_kernel(
    const float* input,
    const float* weight,
    const float* bias,   // can be NULL if no bias
    float* output,
    int N, int C, int H, int W,
    int K, int kH, int kW,
    int outH, int outW,
    int use_bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K * outH * outW;
    // grid-stride loop over output elements
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        int ow = i % outW;
        int oh = (i / outW) % outH;
        int k = (i / (outW * outH)) % K;
        int n = i / (outW * outH * K);

        float sum = 0.0f;
        // For each input channel and kernel position
        for (int c = 0; c < C; ++c) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    int h_in = oh + kh;  // padding=0, stride=1
                    int w_in = ow + kw;
                    // Bounds checking (not needed exactly if outH,outW computed properly without padding)
                    // but we check anyway.
                    if (h_in < H && w_in < W) {
                        float in_val = input[((n * C + c) * H + h_in) * W + w_in];
                        float w_val = weight[((k * C + c) * kH + kh) * kW + kw];
                        sum += in_val * w_val;
                    }
                }
            }
        }
        if (use_bias) {
            sum += bias[k];
        }
        output[i] = sum;
    }
}

// Launcher for conv2d_forward.
extern "C" void conv2d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int kH, int kW,
    int outH, int outW,
    int use_bias,
    cudaStream_t stream)
{
    int total = N * K * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    conv2d_direct_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C, H, W, K, kH, kW, outH, outW, use_bias);
}
