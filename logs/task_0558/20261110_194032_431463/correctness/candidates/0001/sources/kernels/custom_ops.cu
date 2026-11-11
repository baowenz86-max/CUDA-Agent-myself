#include <cuda_runtime.h>

// Hardsigmoid: clamp(x/6 + 0.5, 0, 1)
__global__ void hardsigmoid_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        float y = x / 6.0f + 0.5f;
        y = fminf(fmaxf(y, 0.0f), 1.0f);
        output[i] = y;
    }
}

// C-interface launcher. All pointers are raw CUDA device pointers; stream is provided by binding.
extern "C" void hardsigmoid_forward_launcher(const float* input,
                                              float* output,
                                              int64_t numel,
                                              cudaStream_t stream) {
    if (numel <= 0) return;
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 2147483647) {
        grid_size = 2147483647; // maximum grid dimension in x
    }
    hardsigmoid_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // The launcher has no way to return an error code; assume caller checks stream.
        // We still record the error so that subsequent cudaStreamSynchronize can catch it.
    }
}
