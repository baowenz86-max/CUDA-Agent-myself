#include <cuda_runtime.h>

// CUDA kernels for custom ops used by model_new.py
// All launchers are extern "C" for binding from _binding.cpp

// ---------------------------------------------------------------------
// ConvTranspose1d (bias-free) kernel
// Weight shape: (C_in, C_out, kernel_size) as per PyTorch ConvTranspose1d
// Input: (N, C_in, L_in), Output: (N, C_out, L_out)
// L_out = (L_in - 1) * stride - 2*padding + kernel_size + output_padding
// ---------------------------------------------------------------------
__global__ void conv_transpose1d_bias_free_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding)
{
    int total = N * C_out * L_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    for (int linear_idx = idx; linear_idx < total; linear_idx += gridSize) {
        // decompose linear_idx into n, co, lo
        int lo = linear_idx % L_out;
        int tmp = linear_idx / L_out;
        int co = tmp % C_out;
        int n = tmp / C_out;

        float acc = 0.0f;

        // iterate over kernel positions
        for (int k = 0; k < kernel_size; ++k) {
            int candidate = lo + padding - k;
            if (candidate < 0) continue;
            if (candidate % stride != 0) continue;
            int i = candidate / stride;
            if (i >= L_in) continue;

            // iterate over input channels
            for (int ci = 0; ci < C_in; ++ci) {
                float x_val = input[((n * C_in + ci) * L_in) + i];
                // weight shape: (C_in, C_out, kernel_size)
                float w_val = weight[(ci * C_out + co) * kernel_size + k];
                acc += x_val * w_val;
            }
        }

        output[linear_idx] = acc;
    }
}

extern "C" void conv_transpose1d_bias_free_launcher(
    const float* input,
    const float* weight,
    float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding,
    cudaStream_t stream)
{
    int total = N * C_out * L_out;
    int threads = 256;
    int blocks = min(65535, (total + threads - 1) / threads);
    conv_transpose1d_bias_free_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, output,
        N, C_in, C_out, L_in, L_out,
        kernel_size, stride, padding, output_padding);
}

// ---------------------------------------------------------------------
// Clamp kernel: y = min(max(x, min_val), max_val)
// ---------------------------------------------------------------------
__global__ void clamp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int numel,
    float min_val,
    float max_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    float x = input[idx];
    output[idx] = fminf(fmaxf(x, min_val), max_val);
}

extern "C" void clamp_launcher(
    const float* input,
    float* output,
    int numel,
    float min_val,
    float max_val,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    clamp_kernel<<<blocks, threads, 0, stream>>>(input, output, numel, min_val, max_val);
}

// ---------------------------------------------------------------------
// mask_not_equal kernel: mask[i] = (input[i] != mask_val)
// Output is uint8_t (bool) but will be interpreted as bool tensor
// ---------------------------------------------------------------------
__global__ void mask_not_equal_kernel(
    const float* __restrict__ input,
    uint8_t* __restrict__ mask,
    int numel,
    float mask_val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    mask[idx] = (input[idx] != mask_val) ? 1 : 0;
}

extern "C" void mask_not_equal_launcher(
    const float* input,
    uint8_t* mask,
    int numel,
    float mask_val,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    mask_not_equal_kernel<<<blocks, threads, 0, stream>>>(input, mask, numel, mask_val);
}

// ---------------------------------------------------------------------
// masked_select kernel: compactly select elements where mask != 0
// writes to output sequentially, and updates count (int32 or int64)
// We use int64_t for count to match torch indexing, but internal use int
// ---------------------------------------------------------------------
__global__ void masked_select_kernel(
    const float* __restrict__ input,
    const uint8_t* __restrict__ mask,
    float* __restrict__ output,
    long long* __restrict__ count,
    int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    if (mask[idx]) {
        long long pos = atomicAdd((unsigned long long*)count, 1ULL);
        output[pos] = input[idx];
    }
}

extern "C" void masked_select_launcher(
    const float* input,
    const uint8_t* mask,
    float* output,
    long long* count,
    int numel,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = min(65535, (numel + threads - 1) / threads);
    masked_select_kernel<<<blocks, threads, 0, stream>>>(
        input, mask, output, count, numel);
}
