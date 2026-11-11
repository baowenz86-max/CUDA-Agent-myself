#include <cuda_runtime.h>

// Elementwise cosine kernel optimized with float4 vectorization.
// Each thread processes 4 elements at a time when possible.
__global__ void cos_kernel_vec4(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t numel) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    // Vectorized path: process 4 elements per thread
    const int64_t vec4_count = numel / 4;
    for (int64_t i = idx; i < vec4_count; i += stride) {
        const float4 in_vec = reinterpret_cast<const float4*>(input)[i];
        float4 out_vec;
        out_vec.x = __cosf(in_vec.x);
        out_vec.y = __cosf(in_vec.y);
        out_vec.z = __cosf(in_vec.z);
        out_vec.w = __cosf(in_vec.w);
        reinterpret_cast<float4*>(output)[i] = out_vec;
    }

    // Handle remaining elements (numel % 4) with scalar path
    int64_t remainder_start = vec4_count * 4;
    for (int64_t i = remainder_start + idx; i < numel; i += stride) {
        output[i] = __cosf(input[i]);
    }
}

// Launcher for cos_forward operator. No PyTorch dependencies.
// Contract: input/output are contiguous CUDA float32 tensors with same numel.
void cos_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream) {
    if (numel <= 0) return;

    const int threads = 256;
    // Cap grid to avoid excessive blocks, but use grid-stride loop for full coverage.
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    cos_kernel_vec4<<<blocks, threads, 0, stream>>>(input, output, numel);
}
