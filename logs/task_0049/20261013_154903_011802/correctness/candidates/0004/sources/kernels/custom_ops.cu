#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernels for add_channel_bias and round operations.
// ConvTranspose1d is handled by cuDNN in the C++ binding (not this file).

// Adds a per-channel bias to a 3D tensor of shape [B, C, L].
// Each thread processes one element using grid-stride loop.
__global__ void add_channel_bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ bias,
    int total_elements,
    int C,
    int L)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_elements; i += stride) {
        int c = (i / L) % C;
        output[i] = input[i] + bias[c];
    }
}

// Rounds each element to nearest integer, ties away from zero.
__global__ void round_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_elements; i += stride) {
        float val = input[i];
        // Use nearbyintf with FE_TONEAREST? not available in device code.
        // Manual implementation: round half away from zero.
        float truncated = truncf(val);
        float diff = val - truncated;
        if (diff > 0.5f) {
            output[i] = truncated + 1.0f;
        } else if (diff < -0.5f) {
            output[i] = truncated - 1.0f;
        } else {
            output[i] = truncated;
        }
    }
}

// Launcher for add_channel_bias: output = input + bias[channel]
// input, output are contiguous float32 tensors of shape [B, C, L]
// bias is contiguous float32 tensor of shape [C]
// Returns 0 on success, nonzero on error.
extern "C" int add_channel_bias_launcher(
    float* output,
    const float* input,
    const float* bias,
    int B,
    int C,
    int L,
    cudaStream_t stream)
{
    if (L <= 0 || C <= 0 || B <= 0) {
        return 0;  // empty tensor, nothing to do
    }
    int total_elements = B * C * L;
    if (total_elements <= 0) {
        return 0;
    }

    const int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // Cap blocks to avoid excessive grid size for very large tensors,
    // but grid-stride loop handles large sizes; we can limit to a reasonable max.
    const int max_blocks = 65535;  // common limit, but can compute based on device if needed
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }

    add_channel_bias_kernel<<<blocks, threads, 0, stream>>>(
        output, input, bias, total_elements, C, L);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    return 0;
}

// Launcher for round: output = round(input), same shape, in-place possible.
// input, output are contiguous float32 tensors of same shape.
extern "C" int round_launcher(
    float* output,
    const float* input,
    int total_elements,
    cudaStream_t stream)
{
    if (total_elements <= 0) {
        return 0;
    }

    const int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    const int max_blocks = 65535;
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }

    round_kernel<<<blocks, threads, 0, stream>>>(
        output, input, total_elements);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    return 0;
}
