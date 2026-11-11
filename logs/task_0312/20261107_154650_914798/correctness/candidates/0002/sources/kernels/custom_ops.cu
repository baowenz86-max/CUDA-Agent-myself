#include <cuda_runtime.h>

// Conv1d forward kernel (naive implementation)
// Each thread computes one output element: (n, c_out, l_out)
// Input: N x C_in x L_in, weight: C_out x C_in x K, bias: C_out
// stride, padding, dilation
extern "C" __global__ void conv1d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int L_in,
    int C_out, int K,
    int stride, int padding, int dilation,
    int L_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * L_out) return;

    int l_out = idx % L_out;
    int tmp = idx / L_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k = 0; k < K; ++k) {
            int l_in = l_out * stride - padding + k * dilation;
            if (l_in >= 0 && l_in < L_in) {
                int in_idx = ((n * C_in + c_in) * L_in) + l_in;
                int w_idx = ((c_out * C_in + c_in) * K) + k;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }
    if (bias != nullptr) {
        sum += bias[c_out];
    }
    output[idx] = sum;
}

// MaxPool1d forward kernel
// Each thread computes one output element (n, c, l_out)
// kernel_size, stride, padding, dilation
extern "C" __global__ void maxpool1d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int L_in,
    int kernel_size, int stride, int padding, int dilation,
    int L_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * L_out) return;

    int l_out = idx % L_out;
    int tmp = idx / L_out;
    int c = tmp % C;
    int n = tmp / C;

    int start = l_out * stride - padding;
    float max_val = -1e30f;
    for (int k = 0; k < kernel_size; ++k) {
        int l_in = start + k * dilation;
        if (l_in >= 0 && l_in < L_in) {
            int in_idx = ((n * C + c) * L_in) + l_in;
            float val = input[in_idx];
            if (val > max_val) max_val = val;
        }
    }
    output[idx] = max_val;
}

// Launcher functions
extern "C" void conv1d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int L_in,
    int C_out, int K,
    int stride, int padding, int dilation,
    int L_out,
    cudaStream_t stream)
{
    int total = N * C_out * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv1d_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, L_in, C_out, K,
        stride, padding, dilation, L_out);
}

extern "C" void maxpool1d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int L_in,
    int kernel_size, int stride, int padding, int dilation,
    int L_out,
    cudaStream_t stream)
{
    int total = N * C * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    maxpool1d_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, L_in, kernel_size, stride, padding, dilation, L_out);
}
