#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

// Kernel: log_softmax along dim=1 for float32, shape [B, H, W]
// Each thread handles one (b, w) slice, reducing over H.
// Supports arbitrary strides via input stride parameters.
__global__ void log_softmax_dim1_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int B, int H, int W,
    long long in_stride_b, long long in_stride_h, long long in_stride_w,
    long long out_stride_b, long long out_stride_h, long long out_stride_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * W;
    if (idx >= total) return;

    int b = idx / W;
    int w = idx % W;

    const float* base = in + b * in_stride_b + w * in_stride_w;
    float* out_base = out + b * out_stride_b + w * out_stride_w;

    // Step 1: find max over H
    float max_val = -INFINITY;
    for (int h = 0; h < H; ++h) {
        float val = base[h * in_stride_h];
        if (val > max_val) max_val = val;
    }

    // Step 2: compute sum of exp(x - max)
    float sum_exp = 0.0f;
    for (int h = 0; h < H; ++h) {
        float val = base[h * in_stride_h];
        sum_exp += __expf(val - max_val);
    }

    // Step 3: logsumexp = max + log(sum_exp)
    float logsumexp = max_val + __logf(sum_exp);

    // Step 4: write output = x - logsumexp for each h
    for (int h = 0; h < H; ++h) {
        float val = base[h * in_stride_h];
        out_base[h * out_stride_h] = val - logsumexp;
    }
}

// Kernel: min along dim=1 for float32, shape [B, H, W]
__global__ void min_dim1_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int B, int H, int W,
    long long in_stride_b, long long in_stride_h, long long in_stride_w,
    long long out_stride_b, long long out_stride_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * W;
    if (idx >= total) return;

    int b = idx / W;
    int w = idx % W;

    const float* base = in + b * in_stride_b + w * in_stride_w;
    float min_val = INFINITY;
    for (int h = 0; h < H; ++h) {
        float val = base[h * in_stride_h];
        if (val < min_val) min_val = val;
    }

    out[b * out_stride_b + w * out_stride_w] = min_val;
}

// Host launcher for log_softmax_dim1
// Signature expected by binding.cpp (will be implemented later):
//   void log_softmax_dim1_launcher(float* out, const float* in,
//                                   int B, int H, int W,
//                                   long long in_stride_b, long long in_stride_h, long long in_stride_w,
//                                   long long out_stride_b, long long out_stride_h, long long out_stride_w,
//                                   cudaStream_t stream)
extern "C" void log_softmax_dim1_launcher(
    float* out,
    const float* in,
    int B,
    int H,
    int W,
    long long in_stride_b,
    long long in_stride_h,
    long long in_stride_w,
    long long out_stride_b,
    long long out_stride_h,
    long long out_stride_w,
    cudaStream_t stream)
{
    int total = B * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    log_softmax_dim1_kernel<<<blocks, threads, 0, stream>>>(
        out, in, B, H, W,
        in_stride_b, in_stride_h, in_stride_w,
        out_stride_b, out_stride_h, out_stride_w);
}

// Host launcher for min_dim1
extern "C" void min_dim1_launcher(
    float* out,
    const float* in,
    int B,
    int H,
    int W,
    long long in_stride_b,
    long long in_stride_h,
    long long in_stride_w,
    long long out_stride_b,
    long long out_stride_w,
    cudaStream_t stream)
{
    int total = B * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    min_dim1_kernel<<<blocks, threads, 0, stream>>>(
        out, in, B, H, W,
        in_stride_b, in_stride_h, in_stride_w,
        out_stride_b, out_stride_w);
}
