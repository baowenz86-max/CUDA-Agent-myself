#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Forward declarations of launchers (extern "C" for binding.cpp)
extern "C" void sigmoid_forward_launcher(const float* x, float* y, int64_t numel, cudaStream_t stream);
extern "C" void hardtanh_forward_launcher(const float* x, float* y, int64_t numel, cudaStream_t stream);

// Elementwise sigmoid kernel: y = 1 / (1 + exp(-x)), with grid-stride loop
__global__ void sigmoid_forward_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        float val = input[idx];
        // Clamp to avoid overflow in exp
        val = fminf(fmaxf(val, -88.0f), 88.0f);
        output[idx] = 1.0f / (1.0f + __expf(-val));
    }
}

// Elementwise hardtanh kernel: y = clamp(x, -1, 1), with grid-stride loop
__global__ void hardtanh_forward_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        float val = input[idx];
        output[idx] = fminf(fmaxf(val, -1.0f), 1.0f);
    }
}

// Launcher for sigmoid
void sigmoid_forward_launcher(const float* x, float* y, int64_t numel, cudaStream_t stream) {
    const int block_size = 256;
    // Use fixed grid size to handle large numel via grid-stride loop
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535; // cap at max grid size for x dimension
    sigmoid_forward_kernel<<<(int)grid_size, block_size, 0, stream>>>(x, y, numel);
}

// Launcher for hardtanh
void hardtanh_forward_launcher(const float* x, float* y, int64_t numel, cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    hardtanh_forward_kernel<<<(int)grid_size, block_size, 0, stream>>>(x, y, numel);
}
