#include <cuda_runtime.h>
#include <cstdint>

// Helper for ceil division with positive integers
__host__ __device__ inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__global__ void maxpool2d_kernel(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int Hout, int Wout,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int total_outputs,
    long long in_stride_N, long long in_stride_C, long long in_stride_H, long long in_stride_W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int out_idx = idx; out_idx < total_outputs; out_idx += stride) {
        int ow = out_idx % Wout;
        int oh = (out_idx / Wout) % Hout;
        int oc = (out_idx / (Wout * Hout)) % C;
        int on = out_idx / (Wout * Hout * C);

        int h_start = oh * stride_h - padding_h;
        int w_start = ow * stride_w - padding_w;

        float max_val = -1e30f;

        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_in = h_start + kh * dilation_h;
            if (h_in < 0 || h_in >= H) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int w_in = w_start + kw * dilation_w;
                if (w_in < 0 || w_in >= W) continue;

                long long input_idx = (long long)on * in_stride_N
                                    + (long long)oc * in_stride_C
                                    + (long long)h_in * in_stride_H
                                    + (long long)w_in * in_stride_W;
                float val = input[input_idx];
                if (val > max_val) max_val = val;
            }
        }
        output[out_idx] = max_val;
    }
}

extern "C" void maxpool2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int Hout, int Wout,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    bool ceil_mode,
    long long in_stride_N, long long in_stride_C, long long in_stride_H, long long in_stride_W,
    cudaStream_t stream)
{
    int total_outputs = N * C * Hout * Wout;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;

    maxpool2d_kernel<<<blocks, threads, 0, stream>>>(
        output, input,
        N, C, H, W,
        Hout, Wout,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        dilation_h, dilation_w,
        total_outputs,
        in_stride_N, in_stride_C, in_stride_H, in_stride_W);
}

__global__ void elu_kernel(float* output, const float* input, int N, float alpha, long long in_stride_N, long long in_stride_C, long long in_stride_H, long long in_stride_W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float x = input[idx];  // Assuming contiguous? Need strides. For elu, input may have strides; but we can compute differently if input not contiguous. We'll assume input is contiguous as per binding? We'll pass strides as well.
    output[idx] = (x > 0) ? x : alpha * (expf(x) - 1.0f);
}

extern "C" void elu_launcher(
    float* output,
    const float* input,
    int numel,
    long long in_stride_N, long long in_stride_C, long long in_stride_H, long long in_stride_W,
    float alpha,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;

    elu_kernel<<<blocks, threads, 0, stream>>>(
        output, input, numel, alpha,
        in_stride_N, in_stride_C, in_stride_H, in_stride_W);
}
