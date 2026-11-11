// CUDA kernels for custom ops
// This file contains all CUDA kernels including conv_transpose1d and logsumexp.

#include <cuda_runtime.h>
#include <math.h>

// Compute indices for ConvTranspose1d output.
// Input: [N, C_in, L_in], weight: [C_in, C_out, K]
// Output: [N, C_out, L_out] where L_out = (L_in-1)*stride - 2*padding + K + output_padding
// Each thread computes one output element.
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,  // can be nullptr
    float* __restrict__ output,
    int N, int C_in, int C_out, int L_in, int L_out, int K,
    int stride, int padding, int output_padding,
    bool has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (idx >= total) return;

    int l_out = idx % L_out;
    int tmp = idx / L_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k = 0; k < K; ++k) {
            // Input position: l_in = (l_out + padding - k) / stride, if divisible and within [0, L_in)
            int numerator = l_out + padding - k;
            if (numerator < 0) continue;
            if (numerator % stride != 0) continue;
            int l_in = numerator / stride;
            if (l_in >= 0 && l_in < L_in) {
                float input_val = input[((n * C_in + c_in) * L_in) + l_in];
                float weight_val = weight[((c_in * C_out + c_out) * K) + k];
                sum += input_val * weight_val;
            }
        }
    }
    if (has_bias) {
        sum += bias[c_out];
    }
    output[idx] = sum;
}

// Kernel to compute logsumexp along the last dimension.
// input: [B, C, L] (or any shape with last dim L)
// output: [B, C, 1] if keepdim=true, else [B, C]
// Each thread handles one (b, c) pair and reduces over L.
__global__ void logsumexp_last_dim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int L,
    int input_stride_b, int input_stride_c,
    bool keepdim,
    int output_stride_b, int output_stride_c
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C;
    if (idx >= total) return;

    int b = idx / C;
    int c = idx % C;

    const float* data = input + b * input_stride_b + c * input_stride_c;
    float max_val = -1e30f;
    for (int i = 0; i < L; ++i) {
        float v = data[i];
        if (v > max_val) max_val = v;
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < L; ++i) {
        sum_exp += __expf(data[i] - max_val);
    }

    float result = max_val + logf(sum_exp);

    if (keepdim) {
        output[b * output_stride_b + c * output_stride_c] = result;
    } else {
        output[b * C + c] = result;
    }
}

// Launcher for conv_transpose1d
// Inputs and outputs are assumed to be contiguous in memory.
extern "C" void conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int C_out, int L_in, int L_out, int K,
    int stride, int padding, int output_padding,
    bool has_bias,
    cudaStream_t stream
) {
    int total = N * C_out * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, C_out, L_in, L_out, K,
        stride, padding, output_padding,
        has_bias
    );
}

// Launcher for logsumexp along the last dimension (dim = input.dim()-1)
extern "C" void logsumexp_dim_forward_launcher(
    const float* input,
    float* output,
    int B,
    int C,
    int L,
    int input_stride_b,
    int input_stride_c,
    bool keepdim,
    int output_stride_b,
    int output_stride_c,
    cudaStream_t stream
) {
    int total = B * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    logsumexp_last_dim_kernel<<<blocks, threads, 0, stream>>>(
        input, output, B, C, L,
        input_stride_b, input_stride_c,
        keepdim,
        output_stride_b, output_stride_c
    );
}
