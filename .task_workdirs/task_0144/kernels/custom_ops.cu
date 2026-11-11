#include <cuda_runtime.h>

__global__ void batch_norm_inference_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t L,
    int64_t stride_n,
    int64_t stride_c,
    int64_t stride_l
) {
    int64_t total = N * C * L;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += grid_size) {
        int64_t n = i / (C * L);
        int64_t rem = i - n * C * L;
        int64_t c = rem / L;
        int64_t l = rem - c * L;

        int64_t in_index = n * stride_n + c * stride_c + l * stride_l;
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float gamma = (weight != nullptr) ? weight[c] : 1.0f;
        float beta = (bias != nullptr) ? bias[c] : 0.0f;

        y[i] = (x[in_index] - mean) * inv_std * gamma + beta;
    }
}

extern "C" void batch_norm_inference_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t L,
    int64_t stride_n,
    int64_t stride_c,
    int64_t stride_l,
    cudaStream_t stream
) {
    int64_t total = N * C * L;
    if (total == 0) return;

    const int threads = 256;
    int64_t max_blocks = (total + threads - 1) / threads;
    // Cap grid to avoid exceeding maximum grid dimension / improve scheduling
    int64_t blocks = max_blocks < 65535 ? max_blocks : 65535;

    batch_norm_inference_kernel<<<(int)blocks, threads, 0, stream>>>(
        x, y, weight, bias, running_mean, running_var, eps,
        N, C, L, stride_n, stride_c, stride_l
    );
}
