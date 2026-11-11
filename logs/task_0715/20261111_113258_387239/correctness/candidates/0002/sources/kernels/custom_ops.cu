#include <cuda_runtime.h>

// Since binding guarantees contiguous float32 tensors,
// we can use simple 1D elementwise kernels.

__global__ void abs_kernel(float* __restrict__ out, const float* __restrict__ in, long long numel) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        out[i] = fabsf(in[i]);
    }
}

__global__ void prelu_kernel(float* __restrict__ out, const float* __restrict__ in, const float* __restrict__ weight,
                             long long numel, long long channels, long long inner_size) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float val = in[i];
        long long c = (i / inner_size) % channels;
        float w = weight[c];
        out[i] = (val >= 0.0f) ? val : w * val;
    }
}

extern "C" void abs_forward_launcher(float* out, const float* in, long long numel, cudaStream_t stream) {
    int threads = 256;
    long long blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    abs_kernel<<<blocks, threads, 0, stream>>>(out, in, numel);
}

extern "C" void prelu_forward_launcher(float* out, const float* in, const float* weight,
                                       long long numel, long long channels, long long inner_size,
                                       cudaStream_t stream) {
    int threads = 256;
    long long blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    prelu_kernel<<<blocks, threads, 0, stream>>>(out, in, weight, numel, channels, inner_size);
}
