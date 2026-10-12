#include <cuda_runtime.h>

// Tanh forward: output = tanh(input)
__global__ void tanh_forward_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = tanhf(input[i]);
    }
}

// LeakyReLU forward: output = input > 0 ? input : negative_slope * input
__global__ void leaky_relu_forward_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           float negative_slope,
                                           int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float v = input[i];
        output[i] = (v > 0.0f) ? v : (negative_slope * v);
    }
}

// BatchNorm3d inference: output = (input - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
// Input is 5D: N, C, D, H, W
__global__ void batch_norm3d_inference_kernel(const float* __restrict__ input,
                                               float* __restrict__ output,
                                               const float* __restrict__ weight,
                                               const float* __restrict__ bias,
                                               const float* __restrict__ running_mean,
                                               const float* __restrict__ running_var,
                                               float eps,
                                               int64_t N,
                                               int64_t C,
                                               int64_t spatial_size) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total = N * C * spatial_size;
    for (int64_t i = idx; i < total; i += stride) {
        int64_t c = (i / spatial_size) % C;
        float inv_std = rsqrtf(running_var[c] + eps);
        output[i] = (input[i] - running_mean[c]) * inv_std * weight[c] + bias[c];
    }
}

// Launcher functions (extern "C" to avoid name mangling)
extern "C" void tanh_forward_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      cudaStream_t stream) {
    const int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    // Cap blocks to avoid excessive grid size; grid-stride loop handles the rest
    if (blocks > 65535) blocks = 65535;
    tanh_forward_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void leaky_relu_forward_launcher(const float* input,
                                            float* output,
                                            float negative_slope,
                                            int64_t numel,
                                            cudaStream_t stream) {
    const int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    leaky_relu_forward_kernel<<<blocks, threads, 0, stream>>>(input, output, negative_slope, numel);
}

extern "C" void batch_norm3d_inference_launcher(const float* input,
                                                float* output,
                                                const float* weight,
                                                const float* bias,
                                                const float* running_mean,
                                                const float* running_var,
                                                float eps,
                                                int64_t N,
                                                int64_t C,
                                                int64_t spatial_size,
                                                cudaStream_t stream) {
    const int threads = 256;
    int64_t total = N * C * spatial_size;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    batch_norm3d_inference_kernel<<<blocks, threads, 0, stream>>>(
        input, output, weight, bias, running_mean, running_var, eps, N, C, spatial_size);
}
