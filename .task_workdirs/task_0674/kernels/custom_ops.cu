// kernels/custom_ops.cu
// CUDA kernel for BatchNorm2d inference mode.
// Contract: output[n,c,h,w] = (input[n,c,h,w] - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
// Weight and bias are optional; if null, use 1 and 0 respectively.
// Supports contiguous NCHW input. Kernel uses grid-stride loop to handle arbitrary size.

#include <cuda_runtime.h>
#include <math.h>

// Kernel: one thread per element, grid-stride loop
template <typename scalar_t>
__global__ void batchnorm2d_eval_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const scalar_t* __restrict__ running_mean,
    const scalar_t* __restrict__ running_var,
    int N, int C, int H, int W,
    scalar_t eps) {
    int64_t total = (int64_t)N * C * H * W;
    int64_t HW = (int64_t)H * W;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int64_t c = (idx / HW) % C;
        scalar_t mean = running_mean[c];
        scalar_t var = running_var[c];
        scalar_t invstd = rsqrtf(var + eps);
        scalar_t gamma = (weight != nullptr) ? weight[c] : (scalar_t)1.0f;
        scalar_t beta = (bias != nullptr) ? bias[c] : (scalar_t)0.0f;
        output[idx] = (input[idx] - mean) * invstd * gamma + beta;
    }
}

// Host launcher: raw pointers and scalars only, no PyTorch dependencies.
extern "C" void batchnorm2d_eval_launcher(
    float* output,
    const float* input,
    const float* weight,   // may be nullptr
    const float* bias,     // may be nullptr
    const float* running_mean,
    const float* running_var,
    int N, int C, int H, int W,
    float eps,
    cudaStream_t stream) {

    int64_t total = (int64_t)N * C * H * W;
    if (total == 0) return;

    const int threads = 256;
    int64_t max_blocks = (total + threads - 1) / threads;
    // Cap grid to avoid excessive launch overhead; use grid-stride loop
    int blocks = (max_blocks > 65535) ? 65535 : (int)max_blocks;

    batchnorm2d_eval_kernel<float><<<blocks, threads, 0, stream>>>(
        output, input, weight, bias, running_mean, running_var,
        N, C, H, W, eps);

    // Error check (optional but useful; caller may also synchronize later)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Do not throw; just record. The binding should check stream errors.
        // We leave this as no-op for simplicity; binding may use cudaStreamSynchronize.
    }
}
