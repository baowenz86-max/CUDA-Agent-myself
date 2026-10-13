// CUDA kernels and launchers for non-convolution custom ops.
// ConvTranspose1d and Linear will be implemented in C++ using cuDNN/cuBLAS.
// This file only contains raw CUDA kernels and launchers with C linkage.
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// add_bias_forward: out = input + bias broadcast over last dimension
// input: [B, C, L], bias: [C, 1] (contiguous)
// -----------------------------------------------------------------------------
__global__ void add_bias_kernel(const float* __restrict__ input,
                                const float* __restrict__ bias,
                                float* __restrict__ output,
                                int B, int C, int L) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)B * C * L;
    if (idx >= total) return;
    int l = idx % L;
    long long bc = idx / L;
    int c = bc % C;
    output[idx] = input[idx] + bias[c];
}

extern "C" void add_bias_launcher(const float* input,
                                  const float* bias,
                                  float* output,
                                  int B, int C, int L,
                                  cudaStream_t stream) {
    long long total = (long long)B * C * L;
    const int block_size = 256;
    long long grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 0) {
        add_bias_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
            input, bias, output, B, C, L);
    }
}

// -----------------------------------------------------------------------------
// roll_forward: out = roll(input, shifts, dim) specialized for dim=2, shifts=1
// input: [B, C, L], output same shape
// Equivalent to: output[:, :, 0] = input[:, :, L-1], output[:, :, i] = input[:, :, i-1]
// -----------------------------------------------------------------------------
__global__ void roll_last_dim_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int B, int C, int L) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)B * C * L;
    if (idx >= total) return;
    int l = idx % L;
    long long bc = idx / L;
    long long src_idx;
    if (l == 0) {
        src_idx = bc * L + (L - 1);
    } else {
        src_idx = bc * L + (l - 1);
    }
    output[idx] = input[src_idx];
}

extern "C" void roll_launcher(const float* input,
                              float* output,
                              int B, int C, int L,
                              int shifts, int dim,
                              cudaStream_t stream) {
    // Only supports dim=2 (last dim), shifts=1 for this contract.
    if (dim != 2 || shifts != 1) {
        // Invalid configuration: copy input to output as fallback? The contract
        // guarantees these values, so we can ignore or assert.
        // We'll do nothing here; binding should ensure correct args.
        return;
    }
    long long total = (long long)B * C * L;
    const int block_size = 256;
    long long grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 0) {
        roll_last_dim_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
            input, output, B, C, L);
    }
}

// -----------------------------------------------------------------------------
// flatten_forward: output = input.reshape(B, C*L)  (contiguous copy)
// input: [B, C, L] contiguous, output: [B, C*L]
// -----------------------------------------------------------------------------
__global__ void flatten_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               long long total_elements) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        output[idx] = input[idx];
    }
}

extern "C" void flatten_launcher(const float* input,
                                 float* output,
                                 int B, int C, int L,
                                 int start_dim,
                                 cudaStream_t stream) {
    // Only supports start_dim=1, which flattens all dimensions after batch.
    if (start_dim != 1) return;
    long long total = (long long)B * C * L;
    const int block_size = 256;
    long long grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 0) {
        flatten_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
            input, output, total);
    }
}

// -----------------------------------------------------------------------------
// Declarations for launchers implemented in C++ (cuDNN/cuBLAS) future stage.
// These declarations allow binding.cpp to reference them; no definitions here.
// -----------------------------------------------------------------------------
extern "C" void conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream);

extern "C" void linear_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_features,
    int out_features,
    cudaStream_t stream);
