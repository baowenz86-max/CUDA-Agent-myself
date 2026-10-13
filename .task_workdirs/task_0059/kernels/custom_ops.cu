// custom_ops.cu
// Contains CUDA kernels and launchers for custom operations.
// adaptive_avg_pool1d: custom CUDA implementation.
// conv_transpose2d: custom CUDA implementation (since cuDNN unavailable).

#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// adaptive_avg_pool1d
// -----------------------------------------------------------------------------
__global__ void adaptive_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int L,
    int output_size)
{
    int total = N * C * output_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int pos = idx; pos < total; pos += stride) {
        int out_idx = pos % output_size;
        int tmp = pos / output_size;
        int c = tmp % C;
        int n = tmp / C;

        int start = (int)((long long)out_idx * L / output_size);
        int end = (int)(((long long)(out_idx + 1) * L + output_size - 1) / output_size);
        if (end > L) end = L;

        const float* base = input + ((long long)n * C + c) * L;
        float sum = 0.0f;
        for (int k = start; k < end; ++k) {
            sum += base[k];
        }
        float avg = (end > start) ? sum / (float)(end - start) : 0.0f;
        output[pos] = avg;
    }
}

extern "C" void adaptive_avg_pool1d_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int L,
    int output_size,
    cudaStream_t stream)
{
    int total = N * C * output_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) return;
    adaptive_avg_pool1d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, L, output_size);
}

// -----------------------------------------------------------------------------
// conv_transpose2d (custom CUDA kernel)
// -----------------------------------------------------------------------------
// PyTorch ConvTranspose2d can be implemented as a transposed convolution
// (backward data of a convolution). For each output element, we sum over
// input elements that contribute according to stride, padding, output_padding.
//
// Input shape: [N, C_in, H_in, W_in]
// Weight shape: [C_in, C_out, kH, kW]  (PyTorch layout)
// Output shape: [N, C_out, H_out, W_out]
// where H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding
//
// Formula:
// output[n, co, ho, wo] = bias[co] +
//   sum_{ci, kh, kw} input[n, ci, hi, wi] * weight[ci, co, kh, kw]
// where hi = (ho + padding - kh) / stride, if divisible and in range,
//       wi = (wo + padding - kw) / stride, if divisible and in range.
//
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out,
    int H_in, int W_in,
    int H_out, int W_out,
    int kH, int kW,
    int stride, int padding, int output_padding)
{
    int total = N * C_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_total = blockDim.x * gridDim.x;

    for (int pos = idx; pos < total; pos += stride_total) {
        int wo = pos % W_out;
        int tmp = pos / W_out;
        int ho = tmp % H_out;
        tmp /= H_out;
        int co = tmp % C_out;
        int n = tmp / C_out;

        float sum = 0.0f;
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < kH; ++kh) {
                int hi = ho + padding - kh;
                if (hi < 0 || hi % stride != 0) continue;
                hi /= stride;
                if (hi < 0 || hi >= H_in) continue;
                for (int kw = 0; kw < kW; ++kw) {
                    int wi = wo + padding - kw;
                    if (wi < 0 || wi % stride != 0) continue;
                    wi /= stride;
                    if (wi < 0 || wi >= W_in) continue;
                    // input index: [n, ci, hi, wi]
                    float in_val = input[((n * C_in + ci) * H_in + hi) * W_in + wi];
                    // weight index: [ci, co, kh, kw]
                    float w_val = weight[((ci * C_out + co) * kH + kh) * kW + kw];
                    sum += in_val * w_val;
                }
            }
        }
        if (bias != nullptr) {
            sum += bias[co];
        }
        output[pos] = sum;
    }
}

extern "C" void conv_transpose2d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int C_out,
    int H_in, int W_in,
    int H_out, int W_out,
    int kH, int kW,
    int stride, int padding, int output_padding,
    cudaStream_t stream)
{
    int total = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) return;
    conv_transpose2d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, C_out, H_in, W_in, H_out, W_out, kH, kW,
        stride, padding, output_padding);
}
