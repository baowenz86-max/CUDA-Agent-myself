#include <cuda_runtime.h>

// CUDA kernels and launchers for non-convolutional custom ops.
// Conv/ConvTranspose are implemented in C++ via cuDNN, not here.
// This file must remain free of torch/ATen/c10/pybind dependencies.

// ----------------------------------------------------------------
// triu: zero out elements below the diagonal in the last two dims
// Supports arbitrary strides for the 4D input tensor x.
// shape: [N, C, H, W], diagonal = 0 by default.
// ----------------------------------------------------------------
__global__ void triu_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int N, int C, int H, int W,
    long long sN, long long sC, long long sH, long long sW,
    int diagonal) {
    long long total = (long long)N * C * H * W;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        // Decompose linear index into (n, c, h, w)
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;

        // torch.triu semantics: keep elements where w >= h + diagonal
        float val = 0.0f;
        if (w >= h + diagonal) {
            long long src_offset = (long long)n * sN + (long long)c * sC + (long long)h * sH + (long long)w * sW;
            val = in[src_offset];
        }
        out[idx] = val;
    }
}

// Launcher for triu. Input/output are assumed to be float32 CUDA pointers.
// out is pre-allocated contiguous [N,C,H,W] by binding layer.
extern "C" void triu_launcher(
    float* out,
    const float* in,
    int N, int C, int H, int W,
    long long sN, long long sC, long long sH, long long sW,
    int diagonal,
    cudaStream_t stream) {
    long long total = (long long)N * C * H * W;
    if (total <= 0) return;
    const int block = 256;
    long long grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535; // grid-stride handles more
    triu_kernel<<<(int)grid, block, 0, stream>>>(
        out, in, N, C, H, W, sN, sC, sH, sW, diagonal);
}

// ----------------------------------------------------------------
// bias_add: x[b,c,h,w] += bias[c,0,0]
// x is contiguous [N,C,H,W], bias is contiguous [C,1,1]
// ----------------------------------------------------------------
__global__ void bias_add_kernel(
    float* __restrict__ x,
    const float* __restrict__ bias,
    int N, int C, int H, int W) {
    long long total = (long long)N * C * H * W;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        // n = tmp / C; not needed for bias index
        x[idx] += bias[c]; // bias is contiguous [C,1,1], so bias[c] is correct
    }
}

// Launcher for bias_add. x is modified in-place (or binding layer may pass
// a pre-allocated output that is a copy; either way kernel writes into x).
// bias is contiguous [C,1,1], so pointer arithmetic with c is valid.
extern "C" void bias_add_launcher(
    float* x,
    const float* bias,
    int N, int C, int H, int W,
    cudaStream_t stream) {
    long long total = (long long)N * C * H * W;
    if (total <= 0) return;
    const int block = 256;
    long long grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535;
    bias_add_kernel<<<(int)grid, block, 0, stream>>>(x, bias, N, C, H, W);
}
