#include <cuda_runtime.h>
#include <math.h>

// Maximum pooling 2D kernel with arbitrary strides.
// Output shape: [N, C, outH, outW]
// Input shape: [N, C, inH, inW]
// Input strides: in_N_stride, in_C_stride, in_H_stride, in_W_stride
// Output is contiguous, but we can use index calculation from output coordinates.
__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int inH, int inW,
    int outH, int outW,
    int kernelH, int kernelW,
    int strideH, int strideW,
    int paddingH, int paddingW,
    int dilationH, int dilationW,
    long long in_N_stride, long long in_C_stride, long long in_H_stride, long long in_W_stride
) {
    int total = N * C * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int ow = i % outW;
        int tmp = i / outW;
        int oh = tmp % outH;
        tmp /= outH;
        int c = tmp % C;
        int n = tmp / C;

        int start_h = oh * strideH - paddingH;
        int start_w = ow * strideW - paddingW;

        float max_val = -FLT_MAX;
        for (int kh = 0; kh < kernelH; ++kh) {
            int ih = start_h + kh * dilationH;
            if (ih < 0 || ih >= inH) continue;
            for (int kw = 0; kw < kernelW; ++kw) {
                int iw = start_w + kw * dilationW;
                if (iw < 0 || iw >= inW) continue;

                long long input_offset = (long long)n * in_N_stride
                                      + (long long)c * in_C_stride
                                      + (long long)ih * in_H_stride
                                      + (long long)iw * in_W_stride;
                float val = input[input_offset];
                if (val > max_val) max_val = val;
            }
        }
        output[i] = max_val;
    }
}

// ELU kernel. output[i] = x > 0 ? x : alpha * (expf(x) - 1)
__global__ void elu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long numel,
    float alpha
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < numel; i += stride) {
        float x = input[i];
        output[i] = x > 0.0f ? x : alpha * (expf(x) - 1.0f);
    }
}

// Launcher for custom_maxpool2d.
// Signature matches binding expectations: (const float*, float*, int N, int C, int inH, int inW,
//   int outH, int outW, int kernelH, int kernelW, int strideH, int strideW, int paddingH, int paddingW,
//   int dilationH, int dilationW, long long in_N_stride, long long in_C_stride, long long in_H_stride, long long in_W_stride,
//   cudaStream_t stream)
extern "C" void custom_maxpool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int inH, int inW,
    int outH, int outW,
    int kernelH, int kernelW,
    int strideH, int strideW,
    int paddingH, int paddingW,
    int dilationH, int dilationW,
    long long in_N_stride, long long in_C_stride, long long in_H_stride, long long in_W_stride,
    cudaStream_t stream
) {
    int total = N * C * outH * outW;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks < 1) blocks = 1;

    maxpool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, inH, inW,
        outH, outW,
        kernelH, kernelW,
        strideH, strideW,
        paddingH, paddingW,
        dilationH, dilationW,
        in_N_stride, in_C_stride, in_H_stride, in_W_stride
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a library we cannot throw exceptions directly; the binding should check on next cuda op.
        // But we can call cudaDeviceSynchronize? Better to leave as is; binding should handle errors.
    }
}

// Launcher for custom_elu.
extern "C" void custom_elu_launcher(
    const float* input,
    float* output,
    long long numel,
    float alpha,
    cudaStream_t stream
) {
    const int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    if (blocks_ll < 1) blocks_ll = 1;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL; // grid dimension limit
    int blocks = (int)blocks_ll;

    elu_kernel<<<blocks, threads, 0, stream>>>(
        input, output, numel, alpha
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Ignore error; binding should use cudaDeviceSynchronize and check.
    }
}
