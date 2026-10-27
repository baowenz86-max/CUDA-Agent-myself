#include <cuda_runtime.h>

// Elementwise fused sqrt+round kernel for float32 tensors.
// Input values are assumed non-negative (as per contract).
__global__ void fused_sqrt_round_kernel(float* __restrict__ output,
                                        const float* __restrict__ input,
                                        long long numel) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float val = input[i];
        // torch.sqrt then torch.round (round half to even)
        output[i] = roundf(sqrtf(val));
    }
}

// Host launcher: allocates grid and calls kernel on specified CUDA stream.
// No PyTorch dependencies. numel is total number of elements, assumed to match output.
extern "C" void fused_sqrt_round_launcher(float* output,
                                          const float* input,
                                          long long numel,
                                          cudaStream_t stream) {
    const int threads = 256;
    // Calculate number of blocks to cover numel; if numel is huge, use grid-stride loop
    long long blocks_ll = (numel + threads - 1) / threads;
    // Cap blocks to a reasonable maximum (e.g., 1e6) to avoid excessive grid size;
    // grid-stride loop handles remaining elements.
    const long long max_blocks = 1048576; // 2^20, safe for all current GPUs
    int blocks = (int)(blocks_ll < max_blocks ? blocks_ll : max_blocks);
    if (blocks == 0) return; // empty tensor
    fused_sqrt_round_kernel<<<blocks, threads, 0, stream>>>(output, input, numel);
}
