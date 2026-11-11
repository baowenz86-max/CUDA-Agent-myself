#include <cuda_runtime.h>

// Elementwise max with scalar: output[i] = max(input[i], max_val)
__global__ void clamp_max_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int64_t numel,
                                 float max_val) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float v = input[i];
        output[i] = fmaxf(v, max_val);
    }
}

// PReLU: for element at linear index i in shape [N,C,H,W], 
// c = (i / (H*W)) % C, output = x if x>=0 else weight[c]*x
__global__ void prelu_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             int64_t numel,
                             int C,
                             int64_t HW,
                             const float* __restrict__ weight) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float v = input[i];
        int c = static_cast<int>((i / HW) % C);
        float w = weight[c];
        output[i] = (v >= 0.0f) ? v : (w * v);
    }
}

// Launcher for custom_clamp_max: x (float*), max_val (double), stream
extern "C" void custom_clamp_max_launcher(const float* input,
                                          float* output,
                                          int64_t numel,
                                          float max_val,
                                          cudaStream_t stream) {
    int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid limit for large tensors, grid-stride loop handles rest
    clamp_max_kernel<<<blocks, threads, 0, stream>>>(input, output, numel, max_val);
}

// Launcher for custom_prelu: x (float*), weight (float*), output, sizes
extern "C" void custom_prelu_launcher(const float* input,
                                       float* output,
                                       int64_t numel,
                                       int C,
                                       int64_t HW,
                                       const float* weight,
                                       cudaStream_t stream) {
    int threads = 256;
    int blocks = static_cast<int>((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    prelu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel, C, HW, weight);
}
