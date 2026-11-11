#include <cuda_runtime.h>

// CUDA kernel for ReLU forward: output[i] = max(input[i], 0)
__global__ void relu_forward_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    long numel) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;
    for (long i = idx; i < numel; i += stride) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

// Launcher for relu_forward
// input and output are raw pointers to float CUDA memory
// numel is the total number of elements
// stream is the CUDA stream to launch the kernel on
extern "C" void relu_forward_launcher(const float* input,
                                      float* output,
                                      long numel,
                                      cudaStream_t stream) {
    const int block_size = 256;
    // Calculate grid size that covers all elements, capped at 65535 (gridDim.x limit)
    long grid_size_long = (numel + block_size - 1) / block_size;
    int grid_size = (int)(grid_size_long > 65535 ? 65535 : grid_size_long);
    if (grid_size == 0) grid_size = 1; // handle numel=0 case, though launcher shouldn't be called with 0
    relu_forward_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel);
}
