#include <cuda_runtime.h>

__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int L_in, int C_out, int L_out, int K,
    int stride, int padding, int output_padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    int stride_grid = gridDim.x * blockDim.x;

    for (int linear_idx = idx; linear_idx < total; linear_idx += stride_grid) {
        int n = linear_idx / (C_out * L_out);
        int rem = linear_idx % (C_out * L_out);
        int c_out = rem / L_out;
        int l_out = rem % L_out;

        float acc = bias[c_out];

        for (int k = 0; k < K; ++k) {
            int l_in = l_out * stride - padding + k;
            if (l_in >= 0 && l_in < L_in) {
                for (int c_in = 0; c_in < C_in; ++c_in) {
                    float input_val = input[(n * C_in + c_in) * L_in + l_in];
                    float weight_val = weight[(c_in * C_out + c_out) * K + k];
                    acc += input_val * weight_val;
                }
            }
        }

        output[linear_idx] = acc;
    }
}

extern "C" void conv_transpose1d_custom_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int L_in, int C_out, int L_out, int K,
    int stride, int padding, int output_padding,
    cudaStream_t stream) {

    int total = N * C_out * L_out;
    if (total == 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, L_in, C_out, L_out, K,
        stride, padding, output_padding);
}
