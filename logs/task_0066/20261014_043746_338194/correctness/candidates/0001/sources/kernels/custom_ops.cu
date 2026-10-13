#include <cuda_runtime.h>
#include <cstdint>

// Element-wise greater-than-or-equal kernel: output[i] = (input[i] >= threshold)
__global__ void ge_kernel(const float* __restrict__ input,
                          bool* __restrict__ output,
                          int64_t numel,
                          float threshold) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = (input[i] >= threshold);
    }
}

// Element-wise ReLU6 kernel: output[i] = min(max(input[i], 0), 6)
__global__ void relu6_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        val = fminf(fmaxf(val, 0.0f), 6.0f);
        output[i] = val;
    }
}

// Element-wise multiplication kernel: output[i] = input[i] * mask[i]
__global__ void mul_kernel(const float* __restrict__ input,
                           const float* __restrict__ mask,
                           float* __restrict__ output,
                           int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = input[i] * mask[i];
    }
}

// Helper to get a reasonably sized grid
static inline int get_grid_size(int64_t numel, int block_size) {
    // Use enough blocks to cover all elements, but limit to a reasonable max to avoid excessive launch overhead
    int64_t blocks = (numel + block_size - 1) / block_size;
    if (blocks > 65535) {
        blocks = 65535; // gridDim.x limit, but grid-stride loop handles the rest
    }
    return static_cast<int>(blocks);
}

// Launcher for ge_forward
// Output is bool tensor of same shape as input
// threshold is passed as float
// stream is current CUDA stream
extern "C" void ge_forward_launcher(const float* input,
                                    bool* output,
                                    int64_t numel,
                                    float threshold,
                                    cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = get_grid_size(numel, block_size);
    ge_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel, threshold);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Let binding layer handle error; here just ignore or optionally assert
        // We could write to stderr, but keep simple: no throw in .cu launcher
#ifdef DEBUG
        printf("CUDA error in ge_forward_launcher: %s\n", cudaGetErrorString(err));
#endif
    }
}

// Launcher for relu6_forward
// Output is float tensor of same shape as input
extern "C" void relu6_forward_launcher(const float* input,
                                       float* output,
                                       int64_t numel,
                                       cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = get_grid_size(numel, block_size);
    relu6_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
#ifdef DEBUG
        printf("CUDA error in relu6_forward_launcher: %s\n", cudaGetErrorString(err));
#endif
    }
}

// Launcher for mul_forward
// Output is float tensor of same shape as input
// mask must be float tensor of same shape
extern "C" void mul_forward_launcher(const float* input,
                                     const float* mask,
                                     float* output,
                                     int64_t numel,
                                     cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = get_grid_size(numel, block_size);
    mul_kernel<<<grid_size, block_size, 0, stream>>>(input, mask, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
#ifdef DEBUG
        printf("CUDA error in mul_forward_launcher: %s\n", cudaGetErrorString(err));
#endif
    }
}
