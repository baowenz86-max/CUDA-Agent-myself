#include <cuda_runtime.h>

__global__ void hardtanh_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                long long numel,
                                float min_val,
                                float max_val) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float val = input[i];
        val = fminf(fmaxf(val, min_val), max_val);
        output[i] = val;
    }
}

__global__ void div_scalar_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  long long numel,
                                  float divisor) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < numel; i += stride) {
        output[i] = input[i] / divisor;
    }
}

extern "C" void hardtanh_forward_launcher(const float* input,
                                          float* output,
                                          long long numel,
                                          float min_val,
                                          float max_val,
                                          cudaStream_t stream) {
    const int block_size = 256;
    long long blocks_ll = (numel + block_size - 1) / block_size;
    int blocks = (blocks_ll > 65535) ? 65535 : (int)blocks_ll;
    hardtanh_kernel<<<blocks, block_size, 0, stream>>>(input, output, numel, min_val, max_val);
}

extern "C" void div_scalar_forward_launcher(const float* input,
                                            float* output,
                                            long long numel,
                                            float divisor,
                                            cudaStream_t stream) {
    const int block_size = 256;
    long long blocks_ll = (numel + block_size - 1) / block_size;
    int blocks = (blocks_ll > 65535) ? 65535 : (int)blocks_ll;
    div_scalar_kernel<<<blocks, block_size, 0, stream>>>(input, output, numel, divisor);
}
