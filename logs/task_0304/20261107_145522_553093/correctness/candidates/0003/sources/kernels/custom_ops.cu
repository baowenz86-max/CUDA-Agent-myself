#include <cuda_runtime.h>
#include <cstdint>

// Custom CUDA kernels for view_as_real and SyncBatchNorm (eval mode)
//
// view_as_real_forward:
//   Input: complex64 tensor shape [N,C,H] (contiguous)
//   Output: float32 tensor shape [N,C,H,2] (contiguous)
//   Each complex number c = a + bi is copied to output[...,0] = a, output[...,1] = b.
//
// sync_batchnorm_forward:
//   Input: float32 tensor x shape [N,C,H,W]
//   weight, bias: float32 tensor shape [C]
//   running_mean, running_var: float32 tensor shape [C]
//   eps: float scalar
//   Output: float32 tensor same shape as x, normalized using running stats
//   (eval mode): y = ((x - running_mean[c]) / sqrt(running_var[c] + eps)) * weight[c] + bias[c]
//   According to reference semantics, we use batch statistics?
//   The task contract says 'using batch statistics' but reference model in training=false uses running stats.
//   We follow the provided reference_steps and model forward: SyncBatchNorm in eval mode.
//   The launcher receives running_mean/running_var from binding layer (which can access model buffers).
//
// All launchers are extern "C" and use raw pointers and cudaStream_t only.

namespace {

__global__ void view_as_real_kernel(const float2* __restrict__ input,
                                    float* __restrict__ output,
                                    int64_t total_complex) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_complex; i += stride) {
        float2 val = input[i];
        output[2 * i] = val.x;
        output[2 * i + 1] = val.y;
    }
}

// Normalization kernel using running stats
__global__ void apply_batchnorm_running_stats_kernel(const float* __restrict__ x,
                                                     float* __restrict__ y,
                                                     int64_t total,
                                                     int64_t C,
                                                     int64_t plane,
                                                     const float* __restrict__ mean,
                                                     const float* __restrict__ var,
                                                     const float* __restrict__ weight,
                                                     const float* __restrict__ bias,
                                                     float eps) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        int64_t rem = i % (C * plane);
        int c = rem / plane;
        float v = x[i];
        float normalized = (v - mean[c]) / sqrtf(var[c] + eps);
        y[i] = normalized * weight[c] + bias[c];
    }
}

} // namespace

extern "C" {

void view_as_real_forward_launcher(const void* input_complex,
                                   float* output_real,
                                   int64_t total_complex,
                                   cudaStream_t stream) {
    const float2* input = reinterpret_cast<const float2*>(input_complex);
    int threads = 256;
    int blocks = (int)min((total_complex + threads - 1) / threads, (int64_t)65535);
    if (blocks < 1) blocks = 1;
    view_as_real_kernel<<<blocks, threads, 0, stream>>>(input, output_real, total_complex);
}

void sync_batchnorm_forward_launcher(const float* x,
                                     float* y,
                                     const float* weight,
                                     const float* bias,
                                     const float* running_mean,
                                     const float* running_var,
                                     int64_t N,
                                     int64_t C,
                                     int64_t H,
                                     int64_t W,
                                     float eps,
                                     cudaStream_t stream) {
    // Use running_mean and running_var directly in eval mode.
    // No need to compute batch stats if eval mode uses running stats.
    // However, contract semantics say 'using batch statistics' but reference model is training=False,
    // which BatchNorm uses running stats. We follow reference. To satisfy both, we could compute batch stats if
    // running_mean/var are not provided but here they are always provided by binding.
    // We will implement a direct normalization using running_mean and running_var.
    int64_t total = N * C * H * W;
    int threads = 256;
    int blocks = (int)min((total + threads - 1) / threads, (int64_t)65535);
    if (blocks < 1) blocks = 1;
    apply_batchnorm_running_stats_kernel<<<blocks, threads, 0, stream>>>(
        x, y, total, C, H*W, running_mean, running_var, weight, bias, eps);
}

} // extern "C"
