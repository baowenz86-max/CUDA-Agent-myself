#include <cuda_runtime.h>

// Hardtanh: clamp to [-1, 1]
__global__ void hardtanh_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float v = input[i];
        v = fminf(fmaxf(v, -1.0f), 1.0f);
        output[i] = v;
    }
}

// Leaky ReLU with negative_slope = 0.01
__global__ void leaky_relu_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int64_t numel,
                                  float negative_slope) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float v = input[i];
        output[i] = (v > 0.0f) ? v : (v * negative_slope);
    }
}

extern "C" void hardtanh_forward_launcher(const float* input,
                                           float* output,
                                           int64_t numel,
                                           cudaStream_t stream) {
    if (numel == 0) return;
    const int threads = 256;
    const int64_t max_blocks = 65535;
    const int blocks = static_cast<int>(min<int64_t>((numel + threads - 1) / threads, max_blocks));
    hardtanh_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void leaky_relu_forward_launcher(const float* input,
                                             float* output,
                                             int64_t numel,
                                             float negative_slope,
                                             cudaStream_t stream) {
    if (numel == 0) return;
    const int threads = 256;
    const int64_t max_blocks = 65535;
    const int blocks = static_cast<int>(min<int64_t>((numel + threads - 1) / threads, max_blocks));
    leaky_relu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel, negative_slope);
}
