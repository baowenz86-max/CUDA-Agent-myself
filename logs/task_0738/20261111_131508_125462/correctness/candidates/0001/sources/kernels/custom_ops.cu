#include <cuda_runtime.h>

// Grid-stride copy kernel for custom_stack (concatenate x then y along new leading dim)
__global__ void stack_copy_kernel(const float* __restrict__ x,
                                  const float* __restrict__ y,
                                  float* __restrict__ out,
                                  long long n_per_input) {
    long long total = 2 * n_per_input;
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        if (idx < n_per_input) {
            out[idx] = x[idx];
        } else {
            out[idx] = y[idx - n_per_input];
        }
    }
}

// Grid-stride copy kernel for custom_clone
__global__ void clone_copy_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  long long numel) {
    for (long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         idx < numel;
         idx += (long long)gridDim.x * blockDim.x) {
        out[idx] = in[idx];
    }
}

// Launcher for custom_stack: both inputs have size n_per_input, output size 2*n_per_input
extern "C" void custom_stack_launcher(const float* x,
                                     const float* y,
                                     float* out,
                                     long long n_per_input,
                                     cudaStream_t stream) {
    const int block_size = 256;
    long long total = 2 * n_per_input;
    long long blocks_ll = (total + block_size - 1) / block_size;
    int blocks = (blocks_ll > 2147483647LL) ? 2147483647 : (int)blocks_ll;
    stack_copy_kernel<<<blocks, block_size, 0, stream>>>(x, y, out, n_per_input);
}

// Launcher for custom_clone
extern "C" void custom_clone_launcher(const float* in,
                                      float* out,
                                      long long numel,
                                      cudaStream_t stream) {
    const int block_size = 256;
    long long blocks_ll = (numel + block_size - 1) / block_size;
    int blocks = (blocks_ll > 2147483647LL) ? 2147483647 : (int)blocks_ll;
    clone_copy_kernel<<<blocks, block_size, 0, stream>>>(in, out, numel);
}
