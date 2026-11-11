#include <cuda_runtime.h>

// Kernel: sync_batchnorm_forward (eval mode)
// Applies per-channel affine normalization using running mean/var.
// Input: x (N,C,H,W) float32, weight/bias/running_mean/running_var (C) float32
// Output: y (N,C,H,W) float32
__global__ void sync_batchnorm_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ y,
    int N, int C, int H, int W,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    // Compute channel index
    int c = (idx / (H * W)) % C;

    float mean = running_mean[c];
    float var = running_var[c];
    float w = (weight != nullptr) ? weight[c] : 1.0f;
    float b = (bias != nullptr) ? bias[c] : 0.0f;
    float inv_std = rsqrtf(var + eps);

    y[idx] = (x[idx] - mean) * inv_std * w + b;
}

// Launcher for sync_batchnorm_forward
extern "C" void sync_batchnorm_forward_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* y,
    int N, int C, int H, int W,
    float eps,
    cudaStream_t stream
) {
    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    sync_batchnorm_forward_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, bias, running_mean, running_var, y, N, C, H, W, eps
    );
}

// Kernel: isin_forward
// Checks for exact equality with any element in elements.
// Input: x (same shape as output) float32, elements (K) float32
// Output: mask bool (same shape as x)
__global__ void isin_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ elements,
    bool* __restrict__ mask,
    int total,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float val = x[idx];
    bool found = false;
    for (int k = 0; k < K; ++k) {
        if (val == elements[k]) {
            found = true;
            break;
        }
    }
    mask[idx] = found;
}

// Launcher for isin_forward
extern "C" void isin_forward_launcher(
    const float* x,
    const float* elements,
    bool* mask,
    int total,
    int K,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    isin_forward_kernel<<<blocks, threads, 0, stream>>>(
        x, elements, mask, total, K
    );
}
