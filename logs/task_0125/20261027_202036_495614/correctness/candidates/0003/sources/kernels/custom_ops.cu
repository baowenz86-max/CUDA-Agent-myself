#include <cuda_runtime.h>

// BatchNorm2d eval kernel
__global__ void batchnorm2d_eval_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    int N, int C, int L,
    float eps,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    int total = N * C * L;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int l = i % L;
        int tmp = i / L;
        int c = tmp % C;
        int n = tmp / C;
        int h = 0;
        long long in_offset = n * in_stride_n + c * in_stride_c + h * in_stride_h + l * in_stride_w;
        long long out_offset = n * out_stride_n + c * out_stride_c + h * out_stride_h + l * out_stride_w;
        float x = input[in_offset];
        float scale = weight[c] / sqrtf(running_var[c] + eps);
        float shift = bias[c] - running_mean[c] * scale;
        output[out_offset] = x * scale + shift;
    }
}

extern "C" void batchnorm2d_eval_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int N, int C, int L,
    float eps,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    int total = N * C * L;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    batchnorm2d_eval_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, running_mean, running_var, output,
        N, C, L, eps,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}

// ConvTranspose1d kernel: output[n, co, l] = sum_{ci, k} input[n, ci, floor((l + padding - k) / stride)] * weight[ci, co, k]
// where the index is valid if (l + padding - k) % stride == 0 and 0 <= floor((l + padding - k) / stride) < L_in
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int L_in, int L_out, int K,
    int stride, int padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (idx >= total) return;
    int l = idx % L_out;
    int tmp = idx / L_out;
    int co = tmp % C_out;
    int n = tmp / C_out;
    float sum = 0.0f;
    for (int ci = 0; ci < C_in; ++ci) {
        for (int k = 0; k < K; ++k) {
            int pos = l + padding - k;
            if (pos >= 0 && pos % stride == 0) {
                int li = pos / stride;
                if (li < L_in) {
                    float in_val = input[(n * C_in + ci) * L_in + li];
                    float w_val = weight[(ci * C_out + co) * K + k];
                    sum += in_val * w_val;
                }
            }
        }
    }
    output[idx] = sum + bias[co];
}

extern "C" void conv_transpose1d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int C_out, int L_in, int L_out, int K,
    int stride, int padding,
    cudaStream_t stream)
{
    int total = N * C_out * L_out;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, C_out, L_in, L_out, K,
        stride, padding);
}
