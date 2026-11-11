#include <cuda_runtime.h>

// CUDA kernel for Local Response Normalization across channel dimension.
// Each thread computes one output element at position (n,c,h,w).
// Input and output are assumed to be 4D float32 tensors with arbitrary strides.
// We compute sum of squares of neighboring channels within [c - size//2, c + size//2]
// and produce: y = x / (k + alpha * sum)^beta
__global__ void lrn_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int stride_n, int stride_c, int stride_h, int stride_w,
    int out_stride_n, int out_stride_c, int out_stride_h, int out_stride_w,
    int size, float alpha, float beta, float k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    // decompose linear index into n,c,h,w
    int n = idx / (C * H * W);
    int rem = idx % (C * H * W);
    int c = rem / (H * W);
    rem = rem % (H * W);
    int h = rem / W;
    int w = rem % W;

    const float* x_base = x + n * stride_n + c * stride_c + h * stride_h + w * stride_w;
    float val = *x_base;

    int half = size / 2;
    int c_start = max(0, c - half);
    int c_end = min(C - 1, c + half);

    float sum_sq = 0.0f;
    for (int cc = c_start; cc <= c_end; ++cc) {
        const float* ptr = x + n * stride_n + cc * stride_c + h * stride_h + w * stride_w;
        float v = *ptr;
        sum_sq += v * v;
    }

    float denom = k + alpha * sum_sq;
    float out = val * powf(denom, -beta);

    float* y_ptr = y + n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w;
    *y_ptr = out;
}

// Launcher function to be called from C++ binding.
// All parameters are raw pointers and scalars.
extern "C" void lrn_forward_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    int stride_n, int stride_c, int stride_h, int stride_w,
    int out_stride_n, int out_stride_c, int out_stride_h, int out_stride_w,
    int size, float alpha, float beta, float k,
    cudaStream_t stream)
{
    int total = N * C * H * W;
    if (total == 0) return;

    // Choose block size.
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    lrn_forward_kernel<<<blocks, threads, 0, stream>>>(
        x, y, N, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        size, alpha, beta, k);
}