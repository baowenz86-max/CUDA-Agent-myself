#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void sigmoid_kernel(float* __restrict__ output,
                               const float* __restrict__ input,
                               int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < numel; i += stride) {
        output[i] = sigmoidf(input[i]);
    }
}

extern "C" void sigmoid_forward_launcher(
    float* output,
    const float* input,
    int numel,
    cudaStream_t stream) {
    const int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    sigmoid_kernel<<<blocks, threads, 0, stream>>>(output, input, numel);
}

__global__ void avg_pool2d_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int N, int C, int H, int W,
                                  int outH, int outW) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C * outH * outW;
    if (out_idx >= total_out) return;

    int ow = out_idx % outW;
    int oh = (out_idx / outW) % outH;
    int c = (out_idx / (outW * outH)) % C;
    int n = out_idx / (outW * outH * C);

    int h_start = oh * 2;
    int w_start = ow * 2;

    float sum = 0.0f;
    for (int kh = 0; kh < 2; ++kh) {
        int h = h_start + kh;
        if (h >= H) continue;
        for (int kw = 0; kw < 2; ++kw) {
            int w = w_start + kw;
            if (w >= W) continue;
            sum += input[((n * C + c) * H + h) * W + w];
        }
    }
    output[out_idx] = sum * 0.25f;
}

extern "C" void avg_pool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    cudaStream_t stream) {
    int total_out = N * C * outH * outW;
    const int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    avg_pool2d_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W, outH, outW);
}

__global__ void flatten_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int N, int C, int H, int W,
                               int sN, int sC, int sH, int sW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    int input_offset = n * sN + c * sC + h * sH + w * sW;
    output[idx] = input[input_offset];
}

extern "C" void flatten_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int sN, int sC, int sH, int sW,
    cudaStream_t stream) {
    int total = N * C * H * W;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    flatten_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W, sN, sC, sH, sW);
}

// Linear kernel not used in binding; kept for possible future use.
__global__ void linear_kernel_v2(const float* __restrict__ input,
                                 const float* __restrict__ weight,
                                 const float* __restrict__ bias,
                                 float* __restrict__ output,
                                 int M, int N, int K) {
    // not used
}

extern "C" void linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int M, int N, int K,
    cudaStream_t stream) {
    // not used
}
