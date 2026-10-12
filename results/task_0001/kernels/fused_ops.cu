#include <cuda_runtime.h>
#include <cmath>

__device__ inline float mish_activation(float x, float beta) {
    float z = x * beta;
    if (z > 20.0f) {
        return x;
    } else {
        return x * tanhf(log1pf(expf(z)));
    }
}

__device__ inline float fused_elem_pow2(float v, float negative_slope, float mish_alpha, float abs_max) {
    v = (v >= 0.0f) ? v : v * negative_slope;
    v = v * v; // pow exponent 2
    v = mish_activation(v, mish_alpha);
    v = fmaxf(v, 0.0f); // relu
    v = fminf(v, abs_max); // clamp max, min already 0
    return v;
}

__device__ inline float fused_elem_generic(float v, float negative_slope, float pow_exponent, float mish_alpha, float abs_max) {
    v = (v >= 0.0f) ? v : v * negative_slope;
    v = powf(v, pow_exponent);
    v = fabsf(v);
    v = mish_activation(v, mish_alpha);
    v = fmaxf(v, 0.0f); // relu
    v = fminf(fmaxf(v, 0.0f), abs_max);
    return v;
}

__global__ void fused_ops_pow2_kernel(float* output, const float* input, int64_t numel, float negative_slope, float mish_alpha, float abs_max) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        float v = input[idx];
        output[idx] = fused_elem_pow2(v, negative_slope, mish_alpha, abs_max);
    }
}

__global__ void fused_ops_generic_kernel(float* output, const float* input, int64_t numel, float negative_slope, float pow_exponent, float mish_alpha, float abs_max) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        float v = input[idx];
        output[idx] = fused_elem_generic(v, negative_slope, pow_exponent, mish_alpha, abs_max);
    }
}

extern "C" void fused_ops_launcher(float* output, const float* input, int64_t numel, float negative_slope, float pow_exponent, float mish_alpha, float abs_max, cudaStream_t stream) {
    const int block_size = 256;
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int num_sms = prop.multiProcessorCount;
    int max_blocks_per_sm = 32;
    int max_blocks = num_sms * max_blocks_per_sm;
    int needed_blocks = (int)((numel + block_size - 1) / block_size);
    int blocks = (needed_blocks < max_blocks) ? needed_blocks : max_blocks;
    if (blocks < 1) blocks = 1;

    if (pow_exponent == 2.0f) {
        fused_ops_pow2_kernel<<<blocks, block_size, 0, stream>>>(output, input, numel, negative_slope, mish_alpha, abs_max);
    } else {
        fused_ops_generic_kernel<<<blocks, block_size, 0, stream>>>(output, input, numel, negative_slope, pow_exponent, mish_alpha, abs_max);
    }
}
