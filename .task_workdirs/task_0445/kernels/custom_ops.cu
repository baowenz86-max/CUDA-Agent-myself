#include <cuda_runtime.h>

// Kernel for absolute value
extern "C" __global__ void absolute_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

// Kernel for sync batch norm eval
// Layout: NCHW, channels = C, spatial = H*W
extern "C" __global__ void sync_batch_norm_eval_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int64_t numel,
    int64_t C,
    int64_t HW) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        // Determine channel index from flattened index: i = ((n*C + c)*H + h)*W + w
        int64_t c = (i / HW) % C;
        float w = weight[c];
        float b = bias[c];
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        output[i] = (input[i] - mean) * inv_std * w + b;
    }
}

// Launcher for absolute
extern "C" void absolute_forward(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream) {
    if (numel <= 0) return;
    const int block_size = 256;
    int64_t blocks = (numel + block_size - 1) / block_size;
    // Cap grid to avoid exceeding maximum grid dimension
    if (blocks > 65535) blocks = 65535;
    absolute_forward_kernel<<<blocks, block_size, 0, stream>>>(
        output, input, numel);
}

// Launcher for sync_batch_norm_eval
extern "C" void sync_batch_norm_eval_forward(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t numel,
    int64_t C,
    int64_t HW,
    cudaStream_t stream) {
    if (numel <= 0) return;
    const int block_size = 256;
    int64_t blocks = (numel + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;
    sync_batch_norm_eval_forward_kernel<<<blocks, block_size, 0, stream>>>(
        output, input, weight, bias, running_mean, running_var, eps, numel, C, HW);
}
