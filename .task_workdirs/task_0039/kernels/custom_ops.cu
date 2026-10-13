// CUDA kernels for custom ops: conv2d_bias_bn (fused conv+bias+bn), avg_pool2d, sigmoid.
// All functions are extern "C" launchers with raw pointers and cudaStream_t.
// No torch/ATen/c10 dependencies.

#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Fused Conv2d (3x3, stride=1, padding=1) + bias add + batch norm eval
// Input: (const float* input, int N, int C_in, int H, int W,
//         const float* weight, // [C_out, C_in, 3, 3]
//         const float* conv_bias, // [C_out] may be null
//         const float* add_bias, // [C_out,1,1]
//         const float* bn_weight, const float* bn_bias,
//         const float* running_mean, const float* running_var,
//         float eps,
//         float* output, int C_out)
// Output shape: same as input H,W (since padding=1, kernel=3, stride=1)
__global__ void conv2d_bias_bn_kernel(
    const float* __restrict__ input,
    int N, int C_in, int H, int W,
    const float* __restrict__ weight, // C_out * C_in * 9
    const float* __restrict__ conv_bias, // C_out (may be null if no bias, but assume present)
    const float* __restrict__ add_bias, // C_out
    const float* __restrict__ bn_weight, const float* __restrict__ bn_bias,
    const float* __restrict__ running_mean, const float* __restrict__ running_var,
    float eps,
    float* __restrict__ output,
    int C_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H * W;
    if (idx >= total) return;

    // Decode linear index into n, c_out, h, w
    int w = idx % W;
    int h = (idx / W) % H;
    int c_out = (idx / (W * H)) % C_out;
    int n = idx / (W * H * C_out);

    float sum = 0.0f;
    // Convolution over C_in and 3x3 kernel with padding=1
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = -1; kh <= 1; ++kh) {
            int ih = h + kh;
            if (ih < 0 || ih >= H) continue;
            for (int kw = -1; kw <= 1; ++kw) {
                int iw = w + kw;
                if (iw < 0 || iw >= W) continue;
                // Input pointer
                const float* in_ptr = input + ((n * C_in + c_in) * H + ih) * W + iw;
                // Weight index: (c_out, c_in, kh+1, kw+1)
                const float* w_ptr = weight + ((c_out * C_in + c_in) * 3 + (kh + 1)) * 3 + (kw + 1);
                sum += (*in_ptr) * (*w_ptr);
            }
        }
    }
    // Add conv bias
    if (conv_bias != nullptr) {
        sum += conv_bias[c_out];
    }
    // Add add_bias (broadcast)
    sum += add_bias[c_out];

    // BatchNorm eval: y = (x - running_mean) / sqrt(running_var + eps) * bn_weight + bn_bias
    float mean = running_mean[c_out];
    float var = running_var[c_out];
    float inv_std = rsqrtf(var + eps);
    float normalized = (sum - mean) * inv_std;
    float result = normalized * bn_weight[c_out] + bn_bias[c_out];

    output[idx] = result;
}

extern "C" void conv2d_bias_bn_launcher(
    const float* input,
    int N, int C_in, int H, int W,
    const float* weight,
    const float* conv_bias,
    const float* add_bias,
    const float* bn_weight, const float* bn_bias,
    const float* running_mean, const float* running_var,
    float eps,
    float* output,
    int C_out,
    cudaStream_t stream)
{
    int total = N * C_out * H * W;
    if (total <= 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv2d_bias_bn_kernel<<<blocks, threads, 0, stream>>>(
        input, N, C_in, H, W,
        weight, conv_bias, add_bias,
        bn_weight, bn_bias, running_mean, running_var, eps,
        output, C_out);
}

// -----------------------------------------------------------------------------
// avg_pool2d kernel (unchanged from previous)
__global__ void avg_pool2d_kernel(
    const float* __restrict__ input,
    int N, int C, int H, int W,
    int input_n_stride, int input_c_stride, int input_h_stride, int input_w_stride,
    int kernel_size, int stride, int padding,
    float* __restrict__ output, int outH, int outW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int c = (idx / (outW * outH)) % C;
    int n = idx / (outW * outH * C);

    float sum = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int ih = oh * stride - padding + kh;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int iw = ow * stride - padding + kw;
            if (iw < 0 || iw >= W) continue;
            const float* in_ptr = input + n * input_n_stride + c * input_c_stride + ih * input_h_stride + iw * input_w_stride;
            sum += *in_ptr;
        }
    }
    float avg = sum / (float)(kernel_size * kernel_size);
    output[idx] = avg;
}

extern "C" void avg_pool2d_launcher(
    const float* input,
    int N, int C, int H, int W,
    int input_n_stride, int input_c_stride, int input_h_stride, int input_w_stride,
    int kernel_size, int stride, int padding,
    float* output, int outH, int outW,
    cudaStream_t stream)
{
    int total = N * C * outH * outW;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    avg_pool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, N, C, H, W,
        input_n_stride, input_c_stride, input_h_stride, input_w_stride,
        kernel_size, stride, padding,
        output, outH, outW);
}

// -----------------------------------------------------------------------------
// sigmoid kernel (unchanged)
__global__ void sigmoid_kernel(const float* __restrict__ input, float* __restrict__ output, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float x = input[idx];
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

extern "C" void sigmoid_launcher(
    const float* input,
    float* output,
    int numel,
    cudaStream_t stream)
{
    if (numel <= 0) return;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}
