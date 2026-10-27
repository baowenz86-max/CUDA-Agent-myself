#include <cuda_runtime.h>

// Fused relu then trunc kernel for float32 tensors.
// Each thread processes multiple elements using grid-stride loop.
__global__ void relu_trunc_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 long long numel) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        float val = input[idx];
        // ReLU: max(0, val)
        val = val > 0.0f ? val : 0.0f;
        // Truncate toward zero
        val = truncf(val);
        output[idx] = val;
    }
}

// C-interface launcher. Assumes input and output are device pointers to float32 data.
// numel may be up to 235929600 (fits in long long).
// stream is the CUDA stream to launch on.
extern "C" void relu_trunc_forward_launcher(const float* input,
                                            float* output,
                                            long long numel,
                                            cudaStream_t stream) {
    const int threads = 256;
    const long long max_blocks_per_grid = 2147483647LL; // actual limit may be lower but grid-stride loop handles it
    long long blocks_ll = (numel + threads - 1) / threads;
    // Clamp to max grid dimension (2^31-1) for safety, grid-stride loop will cover rest.
    int blocks = (int)(blocks_ll < max_blocks_per_grid ? blocks_ll : max_blocks_per_grid);
    if (blocks == 0) return;
    relu_trunc_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}
