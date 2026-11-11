#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// BatchNorm2d inference kernel: y = (x - running_mean[c]) * invstd[c] * weight[c] + bias[c]
// broadcast over channel dim. weight/bias may be nullptr (optional).
// Input is NCHW with arbitrary strides; we compute linear index and derive c.
// Grid-stride loop handles large tensors.
// -----------------------------------------------------------------------------
__global__ void batch_norm2d_inference_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    int64_t N,
    int64_t C,
    int64_t HW,
    float eps
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < N * C * HW; i += stride) {
        int64_t c = (i / HW) % C;
        float mean = running_mean[c];
        float var = running_var[c];
        float invstd = rsqrtf(var + eps);
        float val = (x[i] - mean) * invstd;
        if (weight != nullptr) {
            val *= weight[c];
        }
        if (bias != nullptr) {
            val += bias[c];
        }
        y[i] = val;
    }
}

// -----------------------------------------------------------------------------
// Clamp minimum kernel: y = max(x, min_val)
// -----------------------------------------------------------------------------
__global__ void clamp_min_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    float min_val,
    int64_t numel
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        y[i] = fmaxf(x[i], min_val);
    }
}

// -----------------------------------------------------------------------------
// Launchers (C ABI for binding layer)
// -----------------------------------------------------------------------------

extern "C" void batch_norm2d_inference_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    int64_t N,
    int64_t C,
    int64_t HW,
    float eps,
    cudaStream_t stream
) {
    int64_t total = N * C * HW;
    if (total == 0) return;
    // choose block size 256, grid capped at 65535, grid-stride loop handles rest
    int block = 256;
    int grid = (int)std::min<int64_t>((total + block - 1) / block, 65535);
    batch_norm2d_inference_kernel<<<grid, block, 0, stream>>>(
        x, y, weight, bias, running_mean, running_var, N, C, HW, eps);
}

extern "C" void clamp_min_launcher(
    const float* x,
    float* y,
    float min_val,
    int64_t numel,
    cudaStream_t stream
) {
    if (numel == 0) return;
    int block = 256;
    int grid = (int)std::min<int64_t>((numel + block - 1) / block, 65535);
    clamp_min_kernel<<<grid, block, 0, stream>>>(x, y, min_val, numel);
}
