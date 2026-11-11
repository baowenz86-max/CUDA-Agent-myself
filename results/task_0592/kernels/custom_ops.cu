#include <cuda_runtime.h>

// Kernel: concatenate two tensors along dim=0.
// Assumes both inputs have the same shape and are contiguous.
// output layout: first x1_numel elements from x1, then x2_numel elements from x2.
__global__ void cat_dim0_kernel(const float* __restrict__ x1,
                                const float* __restrict__ x2,
                                float* __restrict__ out,
                                long long x1_numel,
                                long long x2_numel) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < x1_numel + x2_numel; i += stride) {
        if (i < x1_numel) {
            out[i] = x1[i];
        } else {
            out[i] = x2[i - x1_numel];
        }
    }
}

// Kernel: element-wise ceil
__global__ void ceil_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            long long numel) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        output[i] = ceilf(input[i]);
    }
}

// Launcher for concatenation along dim=0.
// Inputs: x1, x2 are device pointers; out is device pointer to allocated output.
// x1_numel, x2_numel are the number of elements in each input.
// stream is the CUDA stream to launch on.
extern "C" void custom_cat_forward_launcher(const float* x1,
                                            const float* x2,
                                            float* out,
                                            long long x1_numel,
                                            long long x2_numel,
                                            cudaStream_t stream) {
    // Simple 1D grid-stride kernel.
    int threads = 256;
    long long total = x1_numel + x2_numel;
    long long blocks_ll = (total + threads - 1) / threads;
    // Cap grid at a reasonable number, grid-stride loop handles the rest.
    int blocks = (blocks_ll > 65535) ? 65535 : (int)blocks_ll;
    if (blocks < 1) blocks = 1;
    cat_dim0_kernel<<<blocks, threads, 0, stream>>>(x1, x2, out, x1_numel, x2_numel);
}

// Launcher for ceil.
extern "C" void custom_ceil_forward_launcher(const float* input,
                                             float* output,
                                             long long numel,
                                             cudaStream_t stream) {
    int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    int blocks = (blocks_ll > 65535) ? 65535 : (int)blocks_ll;
    if (blocks < 1) blocks = 1;
    ceil_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}
