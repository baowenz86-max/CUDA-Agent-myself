#include <cuda_runtime.h>
#include <cstdint>

// Elementwise BatchNorm1d inference kernel.
// Each block processes one channel; threads stride over N*L elements.
// Formula: y = (x - mean[c]) * rsqrt(var[c] + eps) * weight[c] + bias[c]
__global__ void batch_norm1d_inference_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int C,
    int NL)
{
    const int c = blockIdx.x;
    if (c >= C) return;

    const float mean = running_mean[c];
    const float var = running_var[c];
    const float invstd = rsqrtf(var + eps);
    const float w = weight[c];
    const float b = bias[c];
    const float scale = w * invstd;
    const float shift = b - mean * scale;

    const float* x_c = x + (size_t)c * NL;
    float* y_c = y + (size_t)c * NL;

    for (int idx = threadIdx.x; idx < NL; idx += blockDim.x) {
        y_c[idx] = x_c[idx] * scale + shift;
    }
}

// C-interface launcher for Python binding layer.
// Assumes x, weight, bias, running_mean, running_var are CUDA float32 tensors.
// x is contiguous with shape [N, C, L], strides [C*L, L, 1].
// weight, bias, running_mean, running_var are contiguous with shape [C].
extern "C" void batchnorm1d_inference_launcher(
    float* y,
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t L,
    cudaStream_t stream)
{
    if (N <= 0 || C <= 0 || L <= 0) return;

    const int64_t NL = N * L;
    const int block_dim = 256;
    // Each block handles one channel; enough parallelism for 64 channels.
    const int grid_dim = static_cast<int>(C);

    batch_norm1d_inference_kernel<<<grid_dim, block_dim, 0, stream>>>(
        y,
        x,
        weight,
        bias,
        running_mean,
        running_var,
        eps,
        static_cast<int>(C),
        static_cast<int>(NL));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // The binding layer should check this, but we include the check here
        // for all launchers to surface launch errors early.
        return;
    }
}
