// custom_ops.cu - all CUDA kernels and launchers for this model
#include <cuda_runtime.h>
#include <cstdint>

// SiLU kernel: y = x * sigmoid(x)
__global__ void silu_kernel(float* __restrict__ out, const float* __restrict__ in, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        float x = in[i];
        float sig = 1.0f / (1.0f + expf(-x));
        out[i] = x * sig;
    }
}

extern "C" void silu_forward_launcher(
    float* out,
    const float* in,
    int64_t n,
    cudaStream_t stream) {

    if (n <= 0) return;

    const int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    silu_kernel<<<blocks, threads, 0, stream>>>(out, in, n);
}

// Transposed 1D convolution (deconvolution) implemented as a direct CUDA kernel.
// Input: x shape (N, C_in, L_in)
// Weight: w shape (C_in, C_out, K)
// Bias: optional bias shape (C_out)
// Output: y shape (N, C_out, L_out)
// Formula: y[n, co, lo] = bias[co] + sum_{ci, k} x[n, ci, li] * w[ci, co, k]
// where li and lo satisfy: lo = li * stride - padding + k, with 0 <= li < L_in, 0 <= k < K
// and output_padding is handled by extending output size; no extra computation.
// We iterate over output positions and gather contributions.
__global__ void conv_transpose1d_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int64_t batch,
    int64_t in_channels,
    int64_t out_channels,
    int64_t length_in,
    int64_t length_out,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    bool has_bias) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch * out_channels * length_out;
    if (idx >= total) return;

    int64_t n = idx / (out_channels * length_out);
    int64_t rem = idx % (out_channels * length_out);
    int64_t co = rem / length_out;
    int64_t lo = rem % length_out;

    float acc = 0.0f;
    const float* x_n = in + n * in_channels * length_in;
    const float* w_co = weight + co * kernel_size; // weight layout: [in_channels, out_channels, kernel_size]

    // Iterate over input indices that contribute to this output position
    // lo = li * stride - padding + k  =>  li = (lo + padding - k) / stride, must be integer and valid
    for (int64_t k = 0; k < kernel_size; ++k) {
        int64_t numerator = lo + padding - k;
        if (numerator % stride != 0) continue;
        int64_t li = numerator / stride;
        if (li < 0 || li >= length_in) continue;

        for (int64_t ci = 0; ci < in_channels; ++ci) {
            const float* x_ci = x_n + ci * length_in;
            const float* w_ci_co = weight + ci * out_channels * kernel_size + co * kernel_size;
            acc += x_ci[li] * w_ci_co[k];
        }
    }

    if (has_bias) {
        acc += bias[co];
    }
    out[idx] = acc;
}

extern "C" void conv_transpose1d_forward_launcher(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    int64_t batch,
    int64_t in_channels,
    int64_t out_channels,
    int64_t length_in,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    cudaStream_t stream) {

    if (batch <= 0 || in_channels <= 0 || out_channels <= 0 || length_in <= 0 || kernel_size <= 0 || stride <= 0) return;

    // Output length formula: (L_in - 1) * stride - 2 * padding + kernel_size + output_padding
    int64_t length_out = (length_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    if (length_out <= 0) return;

    bool has_bias = (bias != nullptr);

    int64_t total = batch * out_channels * length_out;
    const int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        out, in, weight, bias, batch, in_channels, out_channels,
        length_in, length_out, kernel_size, stride, padding, has_bias);
}
