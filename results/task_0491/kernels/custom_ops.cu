#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel for batch normalization evaluation mode
// y = (x - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
// Input x is assumed contiguous with shape (N, C, H, W), treated as float32
__global__ void batch_norm_2d_eval_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N,
    int C,
    int H,
    int W
) {
    int total = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int c = (i / (H * W)) % C;
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float gamma = (weight != nullptr) ? weight[c] : 1.0f;
        float beta = (bias != nullptr) ? bias[c] : 0.0f;
        y[i] = (x[i] - mean) * inv_std * gamma + beta;
    }
}

// CUDA kernel for clamp operation: min(max(x, min_val), max_val)
__global__ void clamp_forward_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    float min_val,
    float max_val,
    int numel
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < numel; i += stride) {
        float v = x[i];
        v = fminf(fmaxf(v, min_val), max_val);
        y[i] = v;
    }
}

// Host launcher for batch_norm_2d_eval
// Inputs: y, x, weight, bias, running_mean, running_var, eps, N, C, H, W, stream
// weight and bias can be nullptr (indicating absence)
// x and y must be contiguous float32 arrays on device
// y should be preallocated with same size as x
extern "C" void batch_norm_2d_eval_launcher(
    float* y,
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream
) {
    int total = static_cast<int>(N * C * H * W);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Cap blocks to avoid excessive grid dimensions; use grid-stride loop
    if (blocks > 65535) blocks = 65535;
    batch_norm_2d_eval_kernel<<<blocks, threads, 0, stream>>>(
        y, x, weight, bias, running_mean, running_var,
        static_cast<float>(eps),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W)
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error handling is left to the binding layer; we don't include headers for logging
    }
}

// Host launcher for clamp_forward
// y, x contiguous float32 arrays on device, numel = number of elements
extern "C" void clamp_forward_launcher(
    float* y,
    const float* x,
    double min_val,
    double max_val,
    int64_t numel,
    cudaStream_t stream
) {
    int total = static_cast<int>(numel);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    clamp_forward_kernel<<<blocks, threads, 0, stream>>>(
        y, x, static_cast<float>(min_val), static_cast<float>(max_val), total
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error handling is left to the binding layer
    }
}
