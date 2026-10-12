#include <cuda_runtime.h>
#include <cmath>

// Compute ConvTranspose1d forward for float32.
// Input: x [N, C_in, L_in]
// Weight: weight [C_in, C_out, K]
// Bias: bias [C_out]
// Output: out [N, C_out, L_out]
// Formula: out[n, co, l] = bias[co] + sum_{ci, k} x[n, ci, i] * weight[ci, co, k]
// where i = (l + padding - output_padding - k) / stride, if integer and within [0, L_in-1]
// and k ranges over kernel_size.
// stride, padding, output_padding are given as ints.

__global__ void conv_transpose1d_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding) {

    int total = N * C_out * L_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid = blockDim.x * gridDim.x;

    for (int pos = idx; pos < total; pos += grid) {
        int n = pos / (C_out * L_out);
        int rem = pos % (C_out * L_out);
        int co = rem / L_out;
        int l = rem % L_out;

        float acc = bias[co];

        for (int ci = 0; ci < C_in; ++ci) {
            for (int k = 0; k < kernel_size; ++k) {
                int numerator = l + padding - output_padding - k;
                if (numerator % stride != 0) continue;
                int i = numerator / stride;
                if (i >= 0 && i < L_in) {
                    acc += x[(n * C_in + ci) * L_in + i] *
                           weight[(ci * C_out + co) * kernel_size + k];
                }
            }
        }

        out[pos] = acc;
    }
}

__global__ void min_scalar_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int numel,
    float scalar) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid = blockDim.x * gridDim.x;
    for (int i = idx; i < numel; i += grid) {
        output[i] = fminf(input[i], scalar);
    }
}

__global__ void erf_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int numel) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid = blockDim.x * gridDim.x;
    for (int i = idx; i < numel; i += grid) {
        output[i] = erff(input[i]);
    }
}

// Launcher: ConvTranspose1d
// Parameters: raw pointers, dimensions, conv parameters, stream.
extern "C" void conv_transpose1d_custom_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding,
    cudaStream_t stream) {

    if (N <= 0 || C_in <= 0 || C_out <= 0 || L_in <= 0 || L_out <= 0 ||
        kernel_size <= 0 || stride <= 0) {
        return;
    }

    int total = N * C_out * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Cap grid to avoid excessive blocks; grid-stride loop handles remainder.
    if (blocks > 65535) blocks = 65535;

    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, bias, out,
        N, C_in, C_out, L_in, L_out,
        kernel_size, stride, padding, output_padding);
}

// Launcher: min with scalar
extern "C" void min_scalar_custom_launcher(
    const float* input,
    float* output,
    int numel,
    float scalar,
    cudaStream_t stream) {

    if (numel <= 0) return;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    min_scalar_kernel<<<blocks, threads, 0, stream>>>(
        input, output, numel, scalar);
}

// Launcher: erf
extern "C" void erf_custom_launcher(
    const float* input,
    float* output,
    int numel,
    cudaStream_t stream) {

    if (numel <= 0) return;
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    erf_kernel<<<blocks, threads, 0, stream>>>(
        input, output, numel);
}
