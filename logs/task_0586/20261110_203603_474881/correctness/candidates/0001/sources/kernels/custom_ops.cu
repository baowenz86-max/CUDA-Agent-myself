#include <cuda_runtime.h>
#include <math.h>

// Element-wise absolute value kernel
__global__ void abs_kernel(const float* __restrict__ input, float* __restrict__ output, long long numel) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        output[idx] = fabsf(input[idx]);
    }
}

// Element-wise power with integer exponent kernel
__global__ void pow_scalar_int_kernel(const float* __restrict__ input, float* __restrict__ output, long long numel, int exponent) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        float val = input[idx];
        float result = 1.0f;
        int exp = exponent;
        // handle negative exponent by reciprocal at end if needed, but contract exponent is positive
        bool neg_exp = exp < 0;
        if (neg_exp) exp = -exp;
        for (int i = 0; i < exp; ++i) {
            result *= val;
        }
        if (neg_exp) result = 1.0f / result;
        output[idx] = result;
    }
}

// Launcher for abs_forward
extern "C" void abs_forward_launcher(const float* input, float* output, long long numel, cudaStream_t stream) {
    const int block_size = 256;
    // ensure grid not exceed max, but for numel up to 268M, grid ~1M blocks, which is okay (< 2^31-1)
    long long grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL; // clamp to int max
    abs_kernel<<<(int)grid_size, block_size, 0, stream>>>(input, output, numel);
}

// Launcher for pow_scalar_forward
extern "C" void pow_scalar_int_launcher(const float* input, float* output, long long numel, int exponent, cudaStream_t stream) {
    const int block_size = 256;
    long long grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL;
    pow_scalar_int_kernel<<<(int)grid_size, block_size, 0, stream>>>(input, output, numel, exponent);
}
