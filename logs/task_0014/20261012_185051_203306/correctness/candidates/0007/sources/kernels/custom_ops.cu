// kernels/custom_ops.cu
#include <cuda_runtime.h>
#include <cstdint>

// Kernel 1: ConvTranspose1d (no bias)
// x: (N, C_in, L_in) contiguous
// weight: (C_in, C_out, K) contiguous
// out: (N, C_out, L_out) contiguous
// L_out = (L_in - 1) * stride - 2*padding + kernel_size + output_padding
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding)
{
    int total = N * C_out * L_out;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int n = idx / (C_out * L_out);
        int rem = idx % (C_out * L_out);
        int co = rem / L_out;
        int lo = rem % L_out;
        float sum = 0.0f;
        for (int ci = 0; ci < C_in; ++ci) {
            const float* x_ptr = x + n * C_in * L_in + ci * L_in;
            const float* w_ptr = weight + ci * C_out * kernel_size + co * kernel_size;
            for (int k = 0; k < kernel_size; ++k) {
                // Correct mapping: input index i = (lo + padding - k) / stride
                // valid only if divisible by stride and 0 <= i < L_in
                int i_candidate = lo + padding - k;
                if (i_candidate % stride == 0) {
                    int i = i_candidate / stride;
                    if (i >= 0 && i < L_in) {
                        sum += x_ptr[i] * w_ptr[k];
                    }
                }
            }
        }
        out[idx] = sum;
    }
}

// Kernel 2: Clamp (min, max)
__global__ void clamp_kernel(const float* __restrict__ in, float* __restrict__ out, int total, float min_val, float max_val) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        float v = in[idx];
        out[idx] = v < min_val ? min_val : (v > max_val ? max_val : v);
    }
}

// Kernel 3: mask_not_equal (x != mask_val)
__global__ void mask_not_equal_kernel(const float* __restrict__ in, bool* __restrict__ mask, int total, float mask_val) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        mask[idx] = (in[idx] != mask_val);
    }
}

// Kernel 4: masked_select
// input: float array of size total
// mask: bool array of size total
// out: output float array of size at least total
// count: pointer to int in device memory (initialized to 0 by binding)
__global__ void masked_select_kernel(const float* __restrict__ in, const bool* __restrict__ mask, float* __restrict__ out, int total, int* __restrict__ count) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        if (mask[idx]) {
            int pos = atomicAdd(count, 1);
            out[pos] = in[idx];
        }
    }
}

// Launchers: extern "C" so binding.cpp can link.
extern "C" void conv_transpose1d_bias_free_launcher(
    const float* x,
    const float* weight,
    float* out,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding,
    cudaStream_t stream)
{
    int total = N * C_out * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, out, N, C_in, C_out, L_in, L_out, kernel_size, stride, padding, output_padding);
}

extern "C" void clamp_launcher(
    const float* in,
    float* out,
    int total,
    float min_val,
    float max_val,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    clamp_kernel<<<blocks, threads, 0, stream>>>(in, out, total, min_val, max_val);
}

extern "C" void mask_not_equal_launcher(
    const float* in,
    bool* mask,
    int total,
    float mask_val,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mask_not_equal_kernel<<<blocks, threads, 0, stream>>>(in, mask, total, mask_val);
}

extern "C" void masked_select_launcher(
    const float* in,
    const bool* mask,
    float* out,
    int total,
    int* count,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    masked_select_kernel<<<blocks, threads, 0, stream>>>(in, mask, out, total, count);
}
