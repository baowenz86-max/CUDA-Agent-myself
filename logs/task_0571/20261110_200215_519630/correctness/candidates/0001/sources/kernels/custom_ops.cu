#include <cuda_runtime.h>

// Elementwise isnan: output[i] = (input[i] != input[i]) ? 1.0f : 0.0f
__global__ void isnan_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             long long numel) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = (val != val) ? 1.0f : 0.0f;
    }
}

// Elementwise ReLU6: output[i] = min(max(input[i], 0.0f), 6.0f)
__global__ void relu6_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             long long numel) {
    long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float val = input[i];
        float clamped = val < 0.0f ? 0.0f : (val > 6.0f ? 6.0f : val);
        output[i] = clamped;
    }
}

extern "C" void custom_isnan_launcher(float* output,
                                      const float* input,
                                      long long numel,
                                      cudaStream_t stream) {
    if (numel <= 0) return;
    const int threads = 256;
    const int max_blocks = 65535 * 32; // conservative, allows large grid
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (blocks > max_blocks) blocks = max_blocks;
    if (blocks <= 0) blocks = 1;
    isnan_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void custom_relu6_launcher(float* output,
                                       const float* input,
                                       long long numel,
                                       cudaStream_t stream) {
    if (numel <= 0) return;
    const int threads = 256;
    const int max_blocks = 65535 * 32;
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (blocks > max_blocks) blocks = max_blocks;
    if (blocks <= 0) blocks = 1;
    relu6_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}
