#include <cuda_runtime.h>

// Element-wise absolute value kernel
__global__ void abs_kernel(const float* __restrict__ input, float* __restrict__ output, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < n; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

// Element-wise clamp kernel with min and max
__global__ void clamp_kernel(const float* __restrict__ input, float* __restrict__ output,
                             float min_val, float max_val, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < n; i += stride) {
        float val = input[i];
        val = fminf(fmaxf(val, min_val), max_val);
        output[i] = val;
    }
}

// Launcher for custom_abs_forward
extern "C" void custom_abs_forward_launcher(const float* input, float* output, long long n, cudaStream_t stream) {
    const int block_size = 256;
    long long grid_size = (n + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) {
        grid_size = 2147483647LL;
    }
    abs_kernel<<<(int)grid_size, block_size, 0, stream>>>(input, output, n);
}

// Launcher for custom_clamp_forward
extern "C" void custom_clamp_forward_launcher(const float* input, float* output,
                                             double min_value, double max_value, long long n, cudaStream_t stream) {
    const int block_size = 256;
    long long grid_size = (n + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) {
        grid_size = 2147483647LL;
    }
    float min_val = (float)min_value;
    float max_val = (float)max_value;
    clamp_kernel<<<(int)grid_size, block_size, 0, stream>>>(input, output, min_val, max_val, n);
}
