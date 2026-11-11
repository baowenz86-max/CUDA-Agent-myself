// custom_ops.cu - CUDA kernels and launchers for clone_to_cuda and rot90_1_2
// This file contains only raw CUDA kernels and C-style launchers.
// No torch/ATen/c10/pybind dependencies.

#include <cuda_runtime.h>
#include <cstdint>

// Kernel: clone (device-to-device copy, used only if both input and output are on GPU)
// The binding layer may choose to use cudaMemcpy instead, but this kernel is provided for completeness.
__global__ void clone_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        output[idx] = input[idx];
    }
}

// Kernel: rot90 with k=1, dims=(1,2)
// Input shape: [B, C, H, W]
// Output shape: [B, W, C, H]
// Mapping: out[b, w, c, h] = in[b, c, H - 1 - h, w]
__global__ void rot90_1_2_kernel(const float* __restrict__ input, float* __restrict__ output,
                                 int64_t B, int64_t C, int64_t H, int64_t W) {
    int64_t total = B * W * C * H;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    
    for (int64_t i = idx; i < total; i += stride) {
        // Decompose output linear index [B, W, C, H] (contiguous in output)
        int64_t h = i % H;
        int64_t tmp = i / H;
        int64_t c = tmp % C;
        tmp /= C;
        int64_t w = tmp % W;
        int64_t b = tmp / W;
        
        // Input linear index [B, C, H, W] (contiguous in input)
        int64_t in_idx = ((b * C + c) * H + (H - 1 - h)) * W + w;
        output[i] = input[in_idx];
    }
}

// Launcher for clone (device-to-device). Assumes input and output are both on current device.
extern "C" void clone_to_cuda_launcher(const float* input, float* output, int64_t total, cudaStream_t stream) {
    if (total <= 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    // Cap blocks to a reasonable max and use grid-stride loop if needed
    if (blocks > 65535) blocks = 65535;
    clone_kernel<<<blocks, threads, 0, stream>>>(input, output, total);
}

// Launcher for rot90_1_2. Assumes input and output are both on current device and contiguous.
extern "C" void rot90_1_2_launcher(const float* input, float* output,
                                   int64_t B, int64_t C, int64_t H, int64_t W, cudaStream_t stream) {
    int64_t total = B * W * C * H;
    if (total <= 0) return;
    int threads = 256;
    int64_t blocks64 = (total + threads - 1) / threads;
    int blocks = (blocks64 > 65535) ? 65535 : (int)blocks64;
    rot90_1_2_kernel<<<blocks, threads, 0, stream>>>(input, output, B, C, H, W);
}
