#include <cuda_runtime.h>

// Helper for grid-stride loops
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

// ConvTranspose1d bias-free forward kernel
// Weight shape: [in_channels, out_channels, kernel_size]
// Input shape: [N, C_in, L_in]
// Output shape: [N, C_out, L_out]
// L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float* __restrict__ out,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding) {
    int total = N * C_out * L_out;
    CUDA_KERNEL_LOOP(idx, total) {
        int n = idx / (C_out * L_out);
        int rem = idx % (C_out * L_out);
        int co = rem / L_out;
        int lo = rem % L_out;
        float acc = 0.0f;
        for (int ci = 0; ci < C_in; ++ci) {
            const float* x_base = x + n * C_in * L_in + ci * L_in;
            const float* w_base = weight + ci * C_out * kernel_size + co * kernel_size;
            for (int k = 0; k < kernel_size; ++k) {
                int lin = lo + padding - k;
                if (lin % stride != 0) continue;
                int i = lin / stride;
                if (i < 0 || i >= L_in) continue;
                acc += x_base[i] * w_base[k];
            }
        }
        out[idx] = acc;
    }
}

// Clamp kernel
template <typename scalar_t>
__global__ void clamp_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, int64_t N, scalar_t min_val, scalar_t max_val) {
    CUDA_KERNEL_LOOP(i, N) {
        scalar_t val = input[i];
        val = val < min_val ? min_val : (val > max_val ? max_val : val);
        output[i] = val;
    }
}

// Mask not equal kernel: produces bool tensor
__global__ void mask_not_equal_kernel(const float* __restrict__ input, bool* __restrict__ output, int64_t N, float mask_val) {
    CUDA_KERNEL_LOOP(i, N) {
        output[i] = (input[i] != mask_val);
    }
}

// Masked select kernel: emits values where mask is true, compacted
__global__ void masked_select_kernel(
    const float* __restrict__ input,
    const bool* __restrict__ mask,
    float* __restrict__ output,
    int64_t N,
    int64_t* __restrict__ count) {
    int64_t write_idx = 0;
    // Single block sequential scan for simplicity, but N large -> use two-pass? Use one block with grid-stride? Here N up to 16M, one block may be slow but correct.
    // Better: use atomicAdd per block? We'll do simple single-thread sequential? Too slow.
    // Use atomicAdd for global offset, then write.
    // For correctness, we'll use one thread per element and atomicAdd on count.
    // But atomicAdd on int64 is supported.
    // We'll initialize count to 0 before kernel.
    CUDA_KERNEL_LOOP(i, N) {
        if (mask[i]) {
            int64_t pos = atomicAdd(reinterpret_cast<unsigned long long*>(count), 1ULL);
            output[pos] = input[i];
        }
    }
}

// Launcher functions

extern "C" void conv_transpose1d_bias_free_launcher(
    const float* x,
    const float* weight,
    float* out,
    int N,
    int C_in,
    int C_out,
    int L_in,
    int L_out,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream) {
    int total = N * C_out * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, out, N, C_in, C_out, L_in, L_out,
        kernel_size, stride, padding, output_padding);
}

extern "C" void clamp_launcher(
    const float* input,
    float* output,
    int64_t N,
    float min_val,
    float max_val,
    cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    clamp_kernel<float><<<blocks, threads, 0, stream>>>(
        input, output, N, min_val, max_val);
}

extern "C" void mask_not_equal_launcher(
    const float* input,
    bool* output,
    int64_t N,
    float mask_val,
    cudaStream_t stream) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    mask_not_equal_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, mask_val);
}

extern "C" void masked_select_launcher(
    const float* input,
    const bool* mask,
    float* output,
    int64_t N,
    int64_t* count,
    cudaStream_t stream) {
    // Initialize count to zero using cudaMemsetAsync
    cudaMemsetAsync(count, 0, sizeof(int64_t), stream);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    masked_select_kernel<<<blocks, threads, 0, stream>>>(
        input, mask, output, N, count);
}
