#include <cuda_runtime.h>

// Element-wise hardtanh (clamp) for float32 contiguous tensors.
// Assumes input and output are contiguous, so linear index i maps directly
// to input[i] and output[i].
__global__ void hardtanh_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t numel,
                                float min_val,
                                float max_val) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        val = fminf(fmaxf(val, min_val), max_val);
        output[i] = val;
    }
}

// C-interface launcher for hardtanh on CUDA float32 contiguous tensors.
extern "C" void hardtanh_forward_launcher(const float* input,
                                          float* output,
                                          int64_t numel,
                                          float min_val,
                                          float max_val,
                                          cudaStream_t stream) {
    if (numel == 0 || input == nullptr || output == nullptr) {
        return;
    }

    const int threads = 256;
    int64_t blocks_64 = (numel + threads - 1) / threads;
    const int64_t max_blocks = 1048576;
    int blocks = (int)(blocks_64 < max_blocks ? blocks_64 : max_blocks);

    hardtanh_kernel<<<blocks, threads, 0, stream>>>(
        input, output, numel, min_val, max_val);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Leave error for binding to check via cudaGetLastError after return.
    }
}
