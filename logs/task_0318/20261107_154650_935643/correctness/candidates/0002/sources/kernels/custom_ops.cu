#include <cuda_runtime.h>

// BatchNorm2d inference kernel
// input: [N,C,H,W] float32
// output: [N,C,H,W] float32
// weight, bias, running_mean, running_var: [C] float32
// eps: float
__global__ void batch_norm2d_inference_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w_idx = idx % W;
    int h_idx = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    // n not needed for batch norm inference

    float mean = running_mean[c];
    float var = running_var[c];
    float inv_std = rsqrtf(var + eps);

    float x = input[idx];
    float y = (x - mean) * inv_std;

    if (weight != nullptr && bias != nullptr) {
        y = y * weight[c] + bias[c];
    } else if (weight != nullptr) {
        y = y * weight[c];
    } else if (bias != nullptr) {
        y = y + bias[c];
    }

    output[idx] = y;
}

// Hardswish kernel: x * min(max(x+3,0),6) / 6
__global__ void hardswish_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    float x = input[idx];
    float y = x * fminf(fmaxf(x + 3.0f, 0.0f), 6.0f) / 6.0f;
    output[idx] = y;
}

// Launcher for batch_norm2d_forward (inference mode)
extern "C" void batch_norm2d_forward_launcher(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    batch_norm2d_inference_kernel<<<blocks, threads, 0, stream>>>(
        input, output, weight, bias, running_mean, running_var,
        eps, N, C, H, W);
}

// Launcher for hardswish_forward
extern "C" void hardswish_forward_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    hardswish_kernel<<<blocks, threads, 0, stream>>>(
        input, output, total);
}
