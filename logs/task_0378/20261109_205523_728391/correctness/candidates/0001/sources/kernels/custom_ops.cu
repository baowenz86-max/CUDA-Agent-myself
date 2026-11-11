#include <cuda_runtime.h>

// BatchNorm2d evaluation mode kernel.
// Computes: out = (x - running_mean[c]) * inv_std * weight[c] + bias[c]
// where c = (idx / (H*W)) % C for contiguous [N,C,H,W] input.
// weight and bias may be null (treated as 1 and 0 respectively).
__global__ void batch_norm_eval_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    double eps,
    int N,
    int C,
    int HW,
    long total_elements)
{
    long idx = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long stride = static_cast<long>(blockDim.x) * gridDim.x;
    for (; idx < total_elements; idx += stride) {
        int c = (idx / HW) % C;
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + static_cast<float>(eps));
        float w = (weight != nullptr) ? weight[c] : 1.0f;
        float b = (bias != nullptr) ? bias[c] : 0.0f;
        float val = x[idx];
        out[idx] = (val - mean) * inv_std * w + b;
    }
}

// Launcher for batch_norm_eval_forward.
// All tensors must be contiguous, float32, CUDA, and have correct shapes.
// x: [N, C, H, W], weight/bias: [C] or null, running_mean/running_var: [C].
extern "C" void batch_norm_eval_launcher(
    const float* x,
    float* out,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int N,
    int C,
    int H,
    int W,
    cudaStream_t stream)
{
    long HW = static_cast<long>(H) * W;
    long total_elements = static_cast<long>(N) * C * HW;
    if (total_elements == 0) {
        return;
    }
    const int block_size = 256;
    int grid_size = static_cast<int>((total_elements + block_size - 1) / block_size);
    // Limit grid size to avoid exceeding max grid dimension (2^31-1), use grid-stride loop.
    const int max_grid_size = 65535 * 32; // A conservative high value, but grid-stride will cover.
    if (grid_size > max_grid_size) {
        grid_size = max_grid_size;
    }
    batch_norm_eval_kernel<<<grid_size, block_size, 0, stream>>>(
        x, out, weight, bias, running_mean, running_var, eps, N, C, static_cast<int>(HW), total_elements);
}
