#include <cuda_runtime.h>

// Element-wise leaky ReLU: output = x >= 0 ? x : x * negative_slope
__global__ void leaky_relu_kernel(float* output, const float* input, int64_t numel, float negative_slope) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        output[i] = (x >= 0.0f) ? x : x * negative_slope;
    }
}

// Element-wise absolute value: output = |x|
__global__ void abs_kernel(float* output, const float* input, int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

extern "C" void leaky_relu_fwd_launcher(
    float* output,
    const float* input,
    int64_t numel,
    float negative_slope,
    cudaStream_t stream
) {
    if (numel <= 0) return;
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535; // grid-stride handles remaining
    leaky_relu_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        output, input, numel, negative_slope
    );
}

extern "C" void abs_fwd_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream
) {
    if (numel <= 0) return;
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535; // grid-stride handles remaining
    abs_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        output, input, numel
    );
}
