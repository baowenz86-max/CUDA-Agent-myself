#include <cuda_runtime.h>
#include <cstdint>

// Softmax along dim=1 for 4D tensor [N, C, H, W]
// Input and output may have arbitrary strides; kernel uses linear index decomposition.
// Stable softmax: subtract max, exp, sum, divide.
__global__ void softmax_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t stride_out_n,
    int64_t stride_out_c,
    int64_t stride_out_h,
    int64_t stride_out_w,
    int64_t stride_in_n,
    int64_t stride_in_c,
    int64_t stride_in_h,
    int64_t stride_in_w)
{
    int64_t total = N * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < total; linear += step) {
        // Decompose linear index over (n, h, w)
        int64_t w = linear % W;
        int64_t tmp = linear / W;
        int64_t h = tmp % H;
        int64_t n = tmp / H;

        // Compute max along C
        float max_val = -1e30f;
        for (int64_t c = 0; c < C; ++c) {
            int64_t in_idx = n * stride_in_n + c * stride_in_c + h * stride_in_h + w * stride_in_w;
            float v = in[in_idx];
            if (v > max_val) max_val = v;
        }

        // Compute sum of exp(x - max)
        float sum = 0.0f;
        for (int64_t c = 0; c < C; ++c) {
            int64_t in_idx = n * stride_in_n + c * stride_in_c + h * stride_in_h + w * stride_in_w;
            float e = expf(in[in_idx] - max_val);
            sum += e;
        }

        // Normalize and write
        for (int64_t c = 0; c < C; ++c) {
            int64_t in_idx = n * stride_in_n + c * stride_in_c + h * stride_in_h + w * stride_in_w;
            int64_t out_idx = n * stride_out_n + c * stride_out_c + h * stride_out_h + w * stride_out_w;
            float e = expf(in[in_idx] - max_val);
            out[out_idx] = e / sum;
        }
    }
}

// Hardswish elementwise: x * clamp(x + 3, 0, 6) / 6
__global__ void hardswish_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int64_t total,
    int64_t stride_out,
    int64_t stride_in)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += step) {
        float x = in[i * stride_in];
        float t = x + 3.0f;
        t = fmaxf(0.0f, fminf(6.0f, t));
        out[i * stride_out] = x * t / 6.0f;
    }
}

// Launcher for softmax, exported for binding
// dim is assumed to be 1; launcher takes full strides to support non-contiguous.
extern "C" void softmax_launcher(
    float* out,
    const float* in,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t stride_out_n,
    int64_t stride_out_c,
    int64_t stride_out_h,
    int64_t stride_out_w,
    int64_t stride_in_n,
    int64_t stride_in_c,
    int64_t stride_in_h,
    int64_t stride_in_w,
    int dim,
    cudaStream_t stream)
{
    if (dim != 1) {
        // For this contract dim is always 1. If other dims are needed, implement separately.
        // Could fallback to atomic implementation, but not needed.
        return;
    }
    int64_t total = N * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid cap, grid-stride loop handles rest
    softmax_kernel<<<blocks, threads, 0, stream>>>(
        out, in, N, C, H, W,
        stride_out_n, stride_out_c, stride_out_h, stride_out_w,
        stride_in_n, stride_in_c, stride_in_h, stride_in_w);
}

// Launcher for hardswish
extern "C" void hardswish_launcher(
    float* out,
    const float* in,
    int64_t total,
    int64_t stride_out,
    int64_t stride_in,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    hardswish_kernel<<<blocks, threads, 0, stream>>>(
        out, in, total, stride_out, stride_in);
}
