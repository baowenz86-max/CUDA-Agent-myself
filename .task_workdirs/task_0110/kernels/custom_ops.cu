#include <cuda_runtime.h>
#include <cstdint>

// Kernel to clamp tensor element-wise: y = min(max(x, min_val), max_val)
__global__ void clamp_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int64_t total,
    const int64_t* __restrict__ in_strides,
    int ndim,
    float min_val,
    float max_val)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        // compute multi-index from linear idx assuming row-major contiguous layout
        // but with provided strides for input; output assumed contiguous (allocated by binding)
        int64_t rem = idx;
        int64_t in_offset = 0;
        // handle up to 4 dimensions (N,C,H,W)
        for (int d = ndim - 1; d >= 0; --d) {
            int64_t dim = (d == ndim - 1) ? rem : 0; // will compute sizes dynamically not needed here
            // We need shape info; for simplicity compute using provided shape via strides? We have only strides.
            // Instead we infer shape from strides and total? Not exact. So assume 4D with known shapes from total and strides.
            // We'll pass shape info via additional parameters? The launcher will pass shape dims.
            // But kernel signature lacks shape; we'll assume dimensions are inferred by stride pattern.
            // To keep simple, this kernel is for contiguous case where we can directly index.
            // For general strides, we'll have separate kernel with shape parameters.
            // To avoid complexity, we'll have specialized kernel for 4D stride case.
        }
        // Placeholder to avoid unused warning - will be replaced
    }
}

// Specialized 4D clamp kernel with stride support
__global__ void clamp_kernel_4d_strided(
    float* __restrict__ out,
    const float* __restrict__ in,
    int N, int C, int H, int W,
    int64_t out_N_stride, int64_t out_C_stride, int64_t out_H_stride, int64_t out_W_stride,
    int64_t in_N_stride, int64_t in_C_stride, int64_t in_H_stride, int64_t in_W_stride,
    float min_val, float max_val)
{
    int64_t total = (int64_t)N * C * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        int n = idx / (C * H * W);
        int rem = idx % (C * H * W);
        int c = rem / (H * W);
        rem = rem % (H * W);
        int h = rem / W;
        int w = rem % W;
        int64_t in_offset = n * in_N_stride + c * in_C_stride + h * in_H_stride + w * in_W_stride;
        int64_t out_offset = n * out_N_stride + c * out_C_stride + h * out_H_stride + w * out_W_stride;
        float val = in[in_offset];
        val = fminf(fmaxf(val, min_val), max_val);
        out[out_offset] = val;
    }
}

// Specialized 4D BN eval kernel with stride support for input and output
__global__ void bn2d_eval_kernel_4d_strided(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N, int C, int H, int W,
    int64_t out_N_stride, int64_t out_C_stride, int64_t out_H_stride, int64_t out_W_stride,
    int64_t in_N_stride, int64_t in_C_stride, int64_t in_H_stride, int64_t in_W_stride)
{
    int64_t total = (int64_t)N * C * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        int n = idx / (C * H * W);
        int rem = idx % (C * H * W);
        int c = rem / (H * W);
        rem = rem % (H * W);
        int h = rem / W;
        int w = rem % W;
        int64_t in_offset = n * in_N_stride + c * in_C_stride + h * in_H_stride + w * in_W_stride;
        int64_t out_offset = n * out_N_stride + c * out_C_stride + h * out_H_stride + w * out_W_stride;
        float val = in[in_offset];
        float mean = running_mean[c];
        float var = running_var[c];
        float invstd = rsqrtf(var + eps);
        float gamma = weight ? weight[c] : 1.0f;
        float beta = bias ? bias[c] : 0.0f;
        val = gamma * (val - mean) * invstd + beta;
        out[out_offset] = val;
    }
}

// Fused clamp + BN eval kernel (4D, strided)
__global__ void clamp_bn2d_eval_kernel_4d_strided(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    float clamp_min, float clamp_max,
    int N, int C, int H, int W,
    int64_t out_N_stride, int64_t out_C_stride, int64_t out_H_stride, int64_t out_W_stride,
    int64_t in_N_stride, int64_t in_C_stride, int64_t in_H_stride, int64_t in_W_stride)
{
    int64_t total = (int64_t)N * C * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        int n = idx / (C * H * W);
        int rem = idx % (C * H * W);
        int c = rem / (H * W);
        rem = rem % (H * W);
        int h = rem / W;
        int w = rem % W;
        int64_t in_offset = n * in_N_stride + c * in_C_stride + h * in_H_stride + w * in_W_stride;
        int64_t out_offset = n * out_N_stride + c * out_C_stride + h * out_H_stride + w * out_W_stride;
        float val = in[in_offset];
        val = fminf(fmaxf(val, clamp_min), clamp_max);
        float mean = running_mean[c];
        float var = running_var[c];
        float invstd = rsqrtf(var + eps);
        float gamma = weight ? weight[c] : 1.0f;
        float beta = bias ? bias[c] : 0.0f;
        val = gamma * (val - mean) * invstd + beta;
        out[out_offset] = val;
    }
}

// Launchers
extern "C" void clamp_forward_launcher(
    float* out,
    const float* in,
    int N, int C, int H, int W,
    int64_t out_N_stride, int64_t out_C_stride, int64_t out_H_stride, int64_t out_W_stride,
    int64_t in_N_stride, int64_t in_C_stride, int64_t in_H_stride, int64_t in_W_stride,
    float min_val, float max_val,
    cudaStream_t stream)
{
    int64_t total = (int64_t)N * C * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    // cap grid to reasonable max
    if (blocks > 65535) blocks = 65535;
    clamp_kernel_4d_strided<<<blocks, threads, 0, stream>>>(
        out, in, N, C, H, W,
        out_N_stride, out_C_stride, out_H_stride, out_W_stride,
        in_N_stride, in_C_stride, in_H_stride, in_W_stride,
        min_val, max_val);
}

extern "C" void bn2d_eval_forward_launcher(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N, int C, int H, int W,
    int64_t out_N_stride, int64_t out_C_stride, int64_t out_H_stride, int64_t out_W_stride,
    int64_t in_N_stride, int64_t in_C_stride, int64_t in_H_stride, int64_t in_W_stride,
    cudaStream_t stream)
{
    int64_t total = (int64_t)N * C * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    bn2d_eval_kernel_4d_strided<<<blocks, threads, 0, stream>>>(
        out, in, weight, bias, running_mean, running_var, eps,
        N, C, H, W,
        out_N_stride, out_C_stride, out_H_stride, out_W_stride,
        in_N_stride, in_C_stride, in_H_stride, in_W_stride);
}

extern "C" void clamp_bn2d_forward_launcher(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    float clamp_min, float clamp_max,
    int N, int C, int H, int W,
    int64_t out_N_stride, int64_t out_C_stride, int64_t out_H_stride, int64_t out_W_stride,
    int64_t in_N_stride, int64_t in_C_stride, int64_t in_H_stride, int64_t in_W_stride,
    cudaStream_t stream)
{
    int64_t total = (int64_t)N * C * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    clamp_bn2d_eval_kernel_4d_strided<<<blocks, threads, 0, stream>>>(
        out, in, weight, bias, running_mean, running_var, eps, clamp_min, clamp_max,
        N, C, H, W,
        out_N_stride, out_C_stride, out_H_stride, out_W_stride,
        in_N_stride, in_C_stride, in_H_stride, in_W_stride);
}
