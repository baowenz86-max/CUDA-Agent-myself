#include <cuda_runtime.h>
#include <cstdint>

// Kernel for cat_dim1: concatenates two 4D tensors along dim=1 (channel dimension)
// Inputs: a (N, C1, H, W), b (N, C2, H, W), output (N, C1+C2, H, W)
// Output strides are assumed contiguous: n*Cout*H*W + c*H*W + h*W + w
// Input strides may be arbitrary; compute using provided strides.
__global__ void cat_dim1_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out,
    int N, int C1, int C2, int H, int W,
    long long a_stride_n, long long a_stride_c, long long a_stride_h, long long a_stride_w,
    long long b_stride_n, long long b_stride_c, long long b_stride_h, long long b_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int Cout = C1 + C2;
    int plane = H * W;
    long long total = (long long)N * Cout * plane;

    for (long long linear = idx; linear < total; linear += stride) {
        int w = linear % W;
        long long tmp = linear / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % Cout;
        int n = tmp / Cout;

        const float* src;
        long long src_offset;
        if (c < C1) {
            src = a;
            src_offset = n * a_stride_n + c * a_stride_c + h * a_stride_h + w * a_stride_w;
        } else {
            int c2 = c - C1;
            src = b;
            src_offset = n * b_stride_n + c2 * b_stride_c + h * b_stride_h + w * b_stride_w;
        }

        long long out_offset = n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w;
        out[out_offset] = src[src_offset];
    }
}

// Kernel for clone_tensor: copies data from input to output assuming contiguous layout.
__global__ void clone_kernel(const float* __restrict__ src, float* __restrict__ dst, long long numel)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = blockDim.x * (long long)gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        dst[i] = src[i];
    }
}

// Launcher for cat_dim1. Expects pointers to device memory.
extern "C" void cat_dim1_launcher(
    const float* a, const float* b, float* out,
    int N, int C1, int C2, int H, int W,
    long long a_stride_n, long long a_stride_c, long long a_stride_h, long long a_stride_w,
    long long b_stride_n, long long b_stride_c, long long b_stride_h, long long b_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    int Cout = C1 + C2;
    long long total = (long long)N * Cout * H * W;
    int threads = 256;
    int blocks = (int)std::min((total + threads - 1) / threads, (long long)INT_MAX);
    if (blocks == 0) blocks = 1;
    cat_dim1_kernel<<<blocks, threads, 0, stream>>>(
        a, b, out, N, C1, C2, H, W,
        a_stride_n, a_stride_c, a_stride_h, a_stride_w,
        b_stride_n, b_stride_c, b_stride_h, b_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}

// Launcher for clone_tensor. Expects device pointers.
extern "C" void clone_tensor_launcher(const float* src, float* dst, long long numel, cudaStream_t stream)
{
    int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    int blocks = (int)std::min(blocks_ll, (long long)INT_MAX);
    if (blocks == 0) blocks = 1;
    clone_kernel<<<blocks, threads, 0, stream>>>(src, dst, numel);
}
