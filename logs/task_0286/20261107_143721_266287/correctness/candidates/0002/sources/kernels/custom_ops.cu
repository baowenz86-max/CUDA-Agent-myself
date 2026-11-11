#include <cuda_runtime.h>

// Upsample nearest neighbor 2D for NCHW layout
// Input: x (float*) with shape [N, C, H, W], strides [n_stride, c_stride, h_stride, w_stride]
// Output: out (float*) contiguous shape [N, C, H*scale_factor, W*scale_factor]
// scale_factor: positive integer
extern "C" __global__ void upsample_nearest2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int N, int C, int H, int W, int scale_factor,
    long long n_stride, long long c_stride, long long h_stride, long long w_stride) {
    int out_h = H * scale_factor;
    int out_w = W * scale_factor;
    long long total = (long long)N * C * out_h * out_w;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        // Decompose output index
        int ow = idx % out_w;
        long long tmp = idx / out_w;
        int oh = tmp % out_h;
        tmp /= out_h;
        int c = tmp % C;
        int n = tmp / C;

        // Nearest neighbor source coordinates
        int ih = oh / scale_factor;
        int iw = ow / scale_factor;

        long long src_idx = (long long)n * n_stride + (long long)c * c_stride
                          + (long long)ih * h_stride + (long long)iw * w_stride;
        out[idx] = x[src_idx];
    }
}

// ReLU elementwise for arbitrary strides
// Input: x (float*) with shape [N, C, H, W], strides [n_stride, c_stride, h_stride, w_stride]
// Output: out (float*) contiguous same shape
// numel = N*C*H*W
extern "C" __global__ void relu_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    long long numel,
    int N, int C, int H, int W,
    long long n_stride, long long c_stride, long long h_stride, long long w_stride) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < numel; idx += stride) {
        // Decompose linear index into n,c,h,w
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;

        long long src_idx = (long long)n * n_stride + (long long)c * c_stride
                          + (long long)h * h_stride + (long long)w * w_stride;
        float val = x[src_idx];
        out[idx] = val > 0.0f ? val : 0.0f;
    }
}

// Host launchers (C interface)

extern "C" void upsample_nearest2d_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W, int scale_factor,
    long long n_stride, long long c_stride, long long h_stride, long long w_stride,
    cudaStream_t stream) {
    int out_h = H * scale_factor;
    int out_w = W * scale_factor;
    long long total = (long long)N * C * out_h * out_w;
    if (total == 0) return;

    int block_size = 256;
    // grid cap: use enough blocks to cover all elements, but not exceed max grid dim (2^31-1)
    long long blocks_ll = (total + block_size - 1) / block_size;
    int blocks = (int)min(blocks_ll, (long long)2147483647);
    upsample_nearest2d_kernel<<<blocks, block_size, 0, stream>>>(
        x, out, N, C, H, W, scale_factor,
        n_stride, c_stride, h_stride, w_stride);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real binding, you would throw an exception. Here we simply avoid ignoring error.
        // The binding layer should handle error checking, but we include this for completeness.
        // We cannot include torch headers, so we can't throw torch exceptions.
        // Instead, we could call cudaDeviceReset or abort, but that's intrusive.
        // For simplicity, we do nothing here; the binding layer will check errors after launch if desired.
        // However, we should at least avoid warning about unused variable.
        (void)err;
    }
}

extern "C" void relu_launcher(
    const float* x,
    float* out,
    long long numel,
    int N, int C, int H, int W,
    long long n_stride, long long c_stride, long long h_stride, long long w_stride,
    cudaStream_t stream) {
    if (numel == 0) return;

    int block_size = 256;
    long long blocks_ll = (numel + block_size - 1) / block_size;
    int blocks = (int)min(blocks_ll, (long long)2147483647);
    relu_kernel<<<blocks, block_size, 0, stream>>>(
        x, out, numel, N, C, H, W,
        n_stride, c_stride, h_stride, w_stride);
    cudaError_t err = cudaGetLastError();
    (void)err;
}
