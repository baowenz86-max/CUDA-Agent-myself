// CUDA kernels for custom ops: isnan and BatchNorm1d forward (eval mode with running stats)
#include <cuda_runtime.h>
#include <stdint.h>

// Element-wise isnan: output bool as unsigned char (1 if NaN, else 0)
__global__ void isnan_kernel(const float* __restrict__ input, uint8_t* __restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = (uint8_t)(isnan(input[i]) ? 1 : 0);
    }
}

// BatchNorm1d forward for eval mode (training=false) using running statistics.
// Input x shape: [N, C, L], output shape same. weight, bias, running_mean, running_var are per-channel arrays of length C.
// y[n,c,l] = weight[c] * (x[n,c,l] - running_mean[c]) / sqrt(running_var[c] + eps) + bias[c]
// If weight/bias null, use identity scale/shift: scale=1, shift=0.
// running_mean/running_var must be non-null in eval mode; contract guarantees non-null in our calls.
__global__ void batch_norm_eval_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t L,
    float* __restrict__ out
) {
    int64_t total = N * C * L;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        // Compute channel index c = (i / L) % C
        int64_t c = (i / L) % C;
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float scale = (weight != nullptr) ? weight[c] : 1.0f;
        float shift = (bias != nullptr) ? bias[c] : 0.0f;
        float x_val = x[i];
        out[i] = scale * (x_val - mean) * inv_std + shift;
    }
}

// Launcher for isnan: input is float32, output is bool (stored as uint8_t)
// The binding layer will provide a torch::Tensor of bool type; data_ptr<bool> corresponds to uint8_t in CUDA.
extern "C" void isnan_forward_launcher(
    const float* input,
    uint8_t* output,
    int64_t numel,
    cudaStream_t stream
) {
    if (numel <= 0) return;
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 1048576) blocks = 1048576;
    isnan_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

// Launcher for batch_norm_forward in eval mode.
// N, C, L are dimensions. weight/bias/running_mean/running_var may be null (but not in our contract).
extern "C" void batch_norm_forward_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    float momentum, // unused in eval mode, but kept for signature compatibility
    int64_t N,
    int64_t C,
    int64_t L,
    float* out,
    cudaStream_t stream
) {
    int64_t total = N * C * L;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 1048576) blocks = 1048576;
    batch_norm_eval_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, bias, running_mean, running_var, eps, N, C, L, out);
}
