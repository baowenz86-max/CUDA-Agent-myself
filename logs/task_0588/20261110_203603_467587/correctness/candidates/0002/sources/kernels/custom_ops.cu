// BatchNorm1d evaluation kernel (CUDA)
// Assumes input is float32, contiguous (or we compute index linearly; binding ensures contiguous)
// Output has same shape as input.

#include <cuda_runtime.h>

__global__ void batch_norm1d_eval_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    int N, int C, int L,
    float eps,
    bool has_weight,
    bool has_bias
) {
    int64_t total = static_cast<int64_t>(N) * C * L;
    // grid-stride loop
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        int l = idx % L;
        int64_t tmp = idx / L;
        int c = tmp % C;
        // n = tmp / C (unused, but computed for completeness)
        // int n = tmp / C;

        float val = input[idx];
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float normalized = (val - mean) * inv_std;
        if (has_weight) {
            normalized *= weight[c];
        }
        if (has_bias) {
            normalized += bias[c];
        }
        output[idx] = normalized;
    }
}

// Launcher function, to be called from binding.cpp with raw pointers
// All tensors are CUDA float32.
extern "C" void batch_norm1d_eval_launcher(
    const float* input,
    float* output,
    const float* weight,   // may be nullptr if no affine
    const float* bias,     // may be nullptr if no affine
    const float* running_mean,
    const float* running_var,
    int N, int C, int L,
    float eps,
    cudaStream_t stream
) {
    int64_t total = static_cast<int64_t>(N) * C * L;
    if (total == 0) return;

    // Determine grid size: use enough blocks to cover all elements with a
    // reasonable number of threads per block.
    const int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    // Limit blocks to avoid excessive launch config; grid-stride loop handles rest.
    if (blocks > 65535) blocks = 65535;

    bool has_weight = (weight != nullptr);
    bool has_bias = (bias != nullptr);

    batch_norm1d_eval_kernel<<<blocks, threads, 0, stream>>>(
        input, output, weight, bias, running_mean, running_var,
        N, C, L, eps, has_weight, has_bias
    );

    // Check for launch errors (caller will also check)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // We cannot throw exceptions here, but we can leave error to be checked by binding.
        // Typically binding will call cudaGetLastError after launcher returns.
        // So we do nothing; just leave the error code.
    }
}
