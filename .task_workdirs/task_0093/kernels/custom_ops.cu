#include <cuda_runtime.h>

// CUDA kernel for mul_bias_forward
// input: [N, C, L] float32 contiguous
// bias: [C, 1] float32 contiguous
// output: [N, C, L] float32 contiguous
// For each element: output[n, c, l] = input[n, c, l] * bias[c]
__global__ void mul_bias_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int L,
    long long total_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < total_elements; i += stride) {
        int l = i % L;
        int c = (i / L) % C;
        output[i] = input[i] * bias[c];
    }
}

// Launcher for mul_bias_forward
extern "C" void mul_bias_forward_launcher(
    const float* input,
    const float* bias,
    float* output,
    int N, int C, int L,
    cudaStream_t stream) {
    long long total = (long long)N * C * L;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    if (blocks > 2147483647LL) blocks = 2147483647LL; // max grid dim
    if (blocks == 0) return;
    mul_bias_forward_kernel<<<(int)blocks, threads, 0, stream>>>(
        input, bias, output, N, C, L, total);
}

// CUDA kernel for conv_transpose1d_forward
// Implements ConvTranspose1d with parameters:
// input: [N, C_in, L_in] float32
// weight: [C_in, C_out, K] float32  (PyTorch ConvTranspose1d weight layout)
// bias: [C_out] float32
// output: [N, C_out, L_out] float32
// stride, padding, output_padding as given
// L_out = (L_in - 1) * stride - 2 * padding + K + output_padding
__global__ void conv_transpose1d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int K, int stride, int padding, int output_padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * L_out) return;
    int n = idx / (C_out * L_out);
    int rem = idx % (C_out * L_out);
    int c_out = rem / L_out;
    int l_out = rem % L_out;
    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k = 0; k < K; ++k) {
            // Correct inverse mapping: l_out = l_in * stride - padding + k
            // => l_in = (l_out + padding - k) / stride if divisible
            int l_in = l_out + padding - k;
            if (l_in % stride != 0) continue;
            l_in /= stride;
            if (l_in >= 0 && l_in < L_in) {
                sum += input[n * C_in * L_in + c_in * L_in + l_in] *
                       weight[c_in * C_out * K + c_out * K + k];
            }
        }
    }
    output[idx] = sum + bias[c_out];
}

// Launcher for conv_transpose1d_forward
extern "C" void conv_transpose1d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int K, int stride, int padding, int output_padding,
    cudaStream_t stream) {
    long long total = (long long)N * C_out * L_out;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    if (blocks > 2147483647LL) blocks = 2147483647LL;
    if (blocks == 0) return;
    conv_transpose1d_forward_kernel<<<(int)blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, C_out, L_in, L_out,
        K, stride, padding, output_padding);
}
