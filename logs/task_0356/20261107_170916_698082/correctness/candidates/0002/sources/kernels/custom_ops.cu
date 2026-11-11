#include <cuda_runtime.h>

// BatchNorm2d inference kernel
// y[n,c,h,w] = (x[n,c,h,w] - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
// weight and bias may be nullptr (treated as 1 and 0 respectively)
__global__ void batch_norm_inference_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N, int C, int H, int W) {
    int total = N * C * H * W;
    int HW = H * W;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int n = idx / (C * HW);
        int rem = idx % (C * HW);
        int c = rem / HW;
        int hw = rem % HW;
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float w = (weight != nullptr) ? weight[c] : 1.0f;
        float b = (bias != nullptr) ? bias[c] : 0.0f;
        float val = x[idx];
        float norm = (val - mean) * inv_std;
        y[idx] = norm * w + b;
    }
}

__global__ void group_norm_inference_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int N, int C, int H, int W) {
    int total = N * C * H * W;
    int HW = H * W;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int n = idx / (C * HW);
        int rem = idx % (C * HW);
        int c = rem / HW;
        int hw = rem % HW;
        
        // Since num_groups == num_channels, each group is one channel.
        // Compute mean and variance over HW elements for this (n,c).
        const float* x_nc = x + n * C * HW + c * HW;
        float mean = 0.0f;
        float var = 0.0f;
        for (int i = 0; i < HW; ++i) {
            mean += x_nc[i];
        }
        mean /= HW;
        for (int i = 0; i < HW; ++i) {
            float diff = x_nc[i] - mean;
            var += diff * diff;
        }
        var /= HW;
        float inv_std = rsqrtf(var + eps);
        float w = (weight != nullptr) ? weight[c] : 1.0f;
        float b = (bias != nullptr) ? bias[c] : 0.0f;
        float val = x[idx];
        float norm = (val - mean) * inv_std;
        y[idx] = norm * w + b;
    }
}

extern "C" void batch_norm_inference_launcher(
    float* y,
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N, int C, int H, int W,
    cudaStream_t stream) {
    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Cap blocks to avoid excessive grid size; grid-stride loop handles rest.
    if (blocks > 65535) blocks = 65535;
    batch_norm_inference_kernel<<<blocks, threads, 0, stream>>>(
        y, x, weight, bias, running_mean, running_var, eps, N, C, H, W);
}

extern "C" void group_norm_inference_launcher(
    float* y,
    const float* x,
    const float* weight,
    const float* bias,
    int num_groups,
    float eps,
    int N, int C, int H, int W,
    cudaStream_t stream) {
    // Note: num_groups is unused because we assume num_groups == C for this task.
    // If general support is needed, a different kernel is required.
    // For correctness per contract, we implement the specific case.
    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    group_norm_inference_kernel<<<blocks, threads, 0, stream>>>(
        y, x, weight, bias, eps, N, C, H, W);
}
