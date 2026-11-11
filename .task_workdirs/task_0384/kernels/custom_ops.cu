#include <cuda_runtime.h>

__global__ void norm_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int64_t numel,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t stride_N,
    int64_t stride_C,
    int64_t stride_H,
    int64_t stride_W)
{
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; i < numel; i += stride) {
        int64_t n = i / (C * H * W);
        int64_t rem = i % (C * H * W);
        int64_t c = rem / (H * W);
        int64_t hw = rem % (H * W);
        int64_t h = hw / W;
        int64_t w = hw % W;

        int64_t offset = n * stride_N + c * stride_C + h * stride_H + w * stride_W;
        float val = x[offset];
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float gamma = (weight != nullptr) ? weight[c] : 1.0f;
        float beta = (bias != nullptr) ? bias[c] : 0.0f;
        out[i] = (val - mean) * inv_std * gamma + beta;
    }
}

extern "C" void norm_forward_launcher(
    const float* x,
    float* out,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    const int64_t* x_strides,
    cudaStream_t stream)
{
    int64_t numel = N * C * H * W;
    int64_t stride_N = x_strides[0];
    int64_t stride_C = x_strides[1];
    int64_t stride_H = x_strides[2];
    int64_t stride_W = x_strides[3];

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    norm_forward_kernel<<<blocks, threads, 0, stream>>>(
        x, out, weight, bias, running_mean, running_var,
        eps, numel, C, H, W,
        stride_N, stride_C, stride_H, stride_W);
}

extern "C" void norm_forward_launcher_strides(
    const float* x,
    float* out,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t stride_N,
    int64_t stride_C,
    int64_t stride_H,
    int64_t stride_W,
    cudaStream_t stream)
{
    int64_t numel = N * C * H * W;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    norm_forward_kernel<<<blocks, threads, 0, stream>>>(
        x, out, weight, bias, running_mean, running_var,
        eps, numel, C, H, W,
        stride_N, stride_C, stride_H, stride_W);
}
