#include <cuda_runtime.h>

// CUDA kernel for deep copy: out[i] = in[i]
// Handles arbitrary numel up to int64_t range with grid-stride loop.
__global__ void clone_kernel(float* __restrict__ out, const float* __restrict__ in, int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        out[i] = in[i];
    }
}

// Host launcher: assumes out and in are valid CUDA float pointers, numel > 0,
// stream is the current CUDA stream obtained by the binding layer.
extern "C" void clone_forward_launcher(float* out, const float* in, int64_t numel, cudaStream_t stream) {
    if (numel <= 0) return;
    const int threads = 256;
    // Cap grid to avoid exceeding device limits; grid-stride handles large data.
    int64_t desired_blocks = (numel + threads - 1) / threads;
    int max_blocks_per_dim = 65535; // safe for all CUDA devices
    int blocks = static_cast<int>(desired_blocks < max_blocks_per_dim ? desired_blocks : max_blocks_per_dim);
    clone_kernel<<<blocks, threads, 0, stream>>>(out, in, numel);
}
