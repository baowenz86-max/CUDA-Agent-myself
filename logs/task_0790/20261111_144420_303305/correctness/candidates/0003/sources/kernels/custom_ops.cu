#include <cuda_runtime.h>

// Kernel: sum over dimension 1 (channels) with keepdim=True
// Input: x shape [B, C, H, W], strides [sB, sC, sH, sW]
// Output: out shape [B, 1, H, W] (contiguous)
// We launch one thread per spatial position (B*H*W), each sums C channel values.
__global__ void sum_dim1_keepdim_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int C, int H, int W,
    long long sB, long long sC, long long sH, long long sW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = B * H * W;
    if (idx >= total_spatial) return;

    // Decompose idx into (b, h, w)
    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    int b = tmp / H;

    const float* base = x + (long long)b * sB + (long long)h * sH + (long long)w * sW;
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        sum += base[(long long)c * sC];
    }
    out[idx] = sum;
}

// Kernel: PReLU elementwise
// out = max(0, x) + weight * min(0, x)
__global__ void prelu_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ weight,
    int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float val = x[idx];
    float w = weight[0];
    out[idx] = (val > 0.0f) ? val : (val * w);
}

// Launcher for sum_dim1_keepdim
extern "C" void sum_dim1_keepdim_launcher(
    const float* x,
    float* out,
    int B, int C, int H, int W,
    long long sB, long long sC, long long sH, long long sW,
    cudaStream_t stream) {
    int total_spatial = B * H * W;
    if (total_spatial <= 0) return;
    const int threads = 256;
    int blocks = (total_spatial + threads - 1) / threads;
    sum_dim1_keepdim_kernel<<<blocks, threads, 0, stream>>>(
        x, out, B, C, H, W, sB, sC, sH, sW);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Do not throw here; binding will check after launch
    }
}

// Launcher for prelu
extern "C" void prelu_launcher(
    const float* x,
    float* out,
    const float* weight,
    int total,
    cudaStream_t stream) {
    if (total <= 0) return;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    prelu_kernel<<<blocks, threads, 0, stream>>>(x, out, weight, total);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Do not throw here; binding will check after launch
    }
}
