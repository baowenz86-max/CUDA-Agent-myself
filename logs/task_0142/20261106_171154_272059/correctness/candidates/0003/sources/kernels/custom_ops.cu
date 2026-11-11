// kernels/custom_ops.cu
// CUDA kernels for batchnorm2d_forward, square_forward, avgpool2d_forward
// Only CUDA kernels and raw-pointer launchers. No torch/ATen/pybind.
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// BatchNorm2d (evaluation mode): y = (x - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
// Supports arbitrary strides on x; output has same strides as x.
// Strides passed as value parameters to avoid pointer issues.
// -----------------------------------------------------------------------------
__global__ void batchnorm2d_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int64_t stride_x0, int64_t stride_x1, int64_t stride_x2, int64_t stride_x3,
    int64_t stride_out0, int64_t stride_out1, int64_t stride_out2, int64_t stride_out3)
{
    int64_t total = (int64_t)N * C * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t tmp = i;
        int w = tmp % W; tmp /= W;
        int h = tmp % H; tmp /= H;
        int c = tmp % C; tmp /= C;
        int n = tmp;

        int64_t in_offset = (int64_t)n * stride_x0 + (int64_t)c * stride_x1 + (int64_t)h * stride_x2 + (int64_t)w * stride_x3;
        int64_t out_offset = (int64_t)n * stride_out0 + (int64_t)c * stride_out1 + (int64_t)h * stride_out2 + (int64_t)w * stride_out3;

        float val = x[in_offset];
        float mean = running_mean[c];
        float var = running_var[c];
        float gamma = weight[c];
        float beta = bias[c];
        float inv_std = rsqrtf(var + eps);
        out[out_offset] = (val - mean) * inv_std * gamma + beta;
    }
}

// -----------------------------------------------------------------------------
// Square: y = x * x, elementwise, supports arbitrary strides.
// Strides passed as value parameters.
// -----------------------------------------------------------------------------
__global__ void square_forward_kernel_4d(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int64_t stride_x0, int64_t stride_x1, int64_t stride_x2, int64_t stride_x3,
    int64_t stride_out0, int64_t stride_out1, int64_t stride_out2, int64_t stride_out3)
{
    int64_t total = (int64_t)N * C * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t tmp = i;
        int w = tmp % W; tmp /= W;
        int h = tmp % H; tmp /= H;
        int c = tmp % C; tmp /= C;
        int n = tmp;
        int64_t in_offset = (int64_t)n * stride_x0 + (int64_t)c * stride_x1 + (int64_t)h * stride_x2 + (int64_t)w * stride_x3;
        int64_t out_offset = (int64_t)n * stride_out0 + (int64_t)c * stride_out1 + (int64_t)h * stride_out2 + (int64_t)w * stride_out3;
        float val = x[in_offset];
        out[out_offset] = val * val;
    }
}

// -----------------------------------------------------------------------------
// AvgPool2d: kernel_size, stride, padding. Output is contiguous with standard strides.
// Input strides passed as values.
// -----------------------------------------------------------------------------
__global__ void avgpool2d_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W,
    int kernel_size, int stride, int padding,
    bool count_include_pad,
    int64_t stride_x0, int64_t stride_x1, int64_t stride_x2, int64_t stride_x3)
{
    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;
    int64_t total = (int64_t)N * C * outH * outW;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride_total = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride_total) {
        int64_t tmp = i;
        int ow = tmp % outW; tmp /= outW;
        int oh = tmp % outH; tmp /= outH;
        int c = tmp % C; tmp /= C;
        int n = tmp;

        float sum = 0.0f;
        int count = 0;
        for (int kh = 0; kh < kernel_size; ++kh) {
            int ih = oh * stride - padding + kh;
            if (ih < 0 || ih >= H) {
                if (count_include_pad) {
                    sum += 0.0f;
                    count++;
                }
                continue;
            }
            for (int kw = 0; kw < kernel_size; ++kw) {
                int iw = ow * stride - padding + kw;
                if (iw < 0 || iw >= W) {
                    if (count_include_pad) {
                        sum += 0.0f;
                        count++;
                    }
                    continue;
                }
                int64_t in_offset = (int64_t)n * stride_x0 + (int64_t)c * stride_x1 + (int64_t)ih * stride_x2 + (int64_t)iw * stride_x3;
                sum += x[in_offset];
                count++;
            }
        }
        int64_t out_offset = ((int64_t)n * C + c) * outH * outW + (int64_t)oh * outW + ow;
        out[out_offset] = sum / (float)count;
    }
}

// -----------------------------------------------------------------------------
// Launchers (extern "C" for binding)
// -----------------------------------------------------------------------------

extern "C" void batchnorm2d_forward_cuda(
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    float* out,
    int N, int C, int H, int W,
    int64_t stride_x0, int64_t stride_x1, int64_t stride_x2, int64_t stride_x3,
    int64_t stride_out0, int64_t stride_out1, int64_t stride_out2, int64_t stride_out3,
    cudaStream_t stream)
{
    int64_t total = (int64_t)N * C * H * W;
    int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 1024) grid_size = 1024;

    batchnorm2d_forward_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        x, weight, bias, running_mean, running_var, eps, out,
        N, C, H, W,
        stride_x0, stride_x1, stride_x2, stride_x3,
        stride_out0, stride_out1, stride_out2, stride_out3);
}

extern "C" void square_forward_cuda(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    int64_t stride_x0, int64_t stride_x1, int64_t stride_x2, int64_t stride_x3,
    int64_t stride_out0, int64_t stride_out1, int64_t stride_out2, int64_t stride_out3,
    cudaStream_t stream)
{
    int64_t total = (int64_t)N * C * H * W;
    int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 1024) grid_size = 1024;

    square_forward_kernel_4d<<<(int)grid_size, block_size, 0, stream>>>(
        x, out, N, C, H, W,
        stride_x0, stride_x1, stride_x2, stride_x3,
        stride_out0, stride_out1, stride_out2, stride_out3);
}

extern "C" void avgpool2d_forward_cuda(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    int kernel_size, int stride, int padding,
    bool count_include_pad,
    int64_t stride_x0, int64_t stride_x1, int64_t stride_x2, int64_t stride_x3,
    cudaStream_t stream)
{
    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;
    int64_t total = (int64_t)N * C * outH * outW;
    int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 1024) grid_size = 1024;

    avgpool2d_forward_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        x, out, N, C, H, W, kernel_size, stride, padding, count_include_pad,
        stride_x0, stride_x1, stride_x2, stride_x3);
}
