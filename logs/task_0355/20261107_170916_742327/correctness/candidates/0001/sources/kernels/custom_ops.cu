#include <cuda_runtime.h>

// Batch normalization inference kernel for contiguous NCHW float32 tensor.
// Each thread processes one output element. The channel index is computed from the linear index.
// Since input is contiguous, the formula: idx = ((n * C + c) * H + h) * W + w.
// Here we simply use the linear index and derive channel: c = (idx / (H*W)) % C.
// This works because NCHW contiguous.
__global__ void batch_norm_inference_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int64_t total_elements,
    int64_t HW,
    int C
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        int c = (int)((i / HW) % C);
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        output[i] = (input[i] - mean) * inv_std * weight[c] + bias[c];
    }
}

// Launcher for batch normalization inference.
// All pointers must be valid CUDA device pointers; tensors are assumed contiguous float32.
// N, C, H, W are dimensions.
extern "C" void batch_norm_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream
) {
    int64_t total_elements = N * C * H * W;
    int64_t HW = H * W;
    int threads = 256;
    int64_t blocks = (total_elements + threads - 1) / threads;
    if (blocks > 2147483647) {
        blocks = 2147483647;
    }
    batch_norm_inference_kernel<<<(int)blocks, threads, 0, stream>>>(
        output, input, weight, bias, running_mean, running_var, eps, total_elements, HW, (int)C);
}
