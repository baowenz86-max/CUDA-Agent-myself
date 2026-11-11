#include <cuda_runtime.h>

// ReflectionPad2d kernel for float32, padding=1 on all sides.
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int pad) {
    int total = N * C * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int ow = i % outW;
        int oh = (i / outW) % outH;
        int oc = (i / (outW * outH)) % C;
        int on = i / (outW * outH * C);

        int iw = ow - pad;
        int ih = oh - pad;
        if (iw < 0) iw = -iw;
        if (iw >= W) iw = 2 * W - 2 - iw;
        if (ih < 0) ih = -ih;
        if (ih >= H) ih = 2 * H - 2 - ih;
        if (iw < 0) iw = 0;
        if (ih < 0) ih = 0;
        if (iw >= W) iw = W - 1;
        if (ih >= H) ih = H - 1;

        output[i] = input[((on * C + oc) * H + ih) * W + iw];
    }
}

// Sigmoid elementwise kernel
__global__ void sigmoid_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        float x = input[i];
        output[i] = 1.0f / (1.0f + expf(-x));
    }
}

// Add bias kernel: broadcasts bias (C,1,1) over input (N,C,H,W).
__global__ void add_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W) {
    int total = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int c = (i / (H * W)) % C;
        output[i] = input[i] + bias[c];
    }
}

// Naive 2D convolution kernel (no cuDNN)
// input: [N, C, H, W], weight: [K, C, R, S], output: [N, K, outH, outW]
// stride, padding, dilation assumed 1,0,1 respectively for this task
__global__ void conv2d_naive_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int outH, int outW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K * outH * outW;
    if (idx >= total) return;

    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int k = (idx / (outW * outH)) % K;
    int n = idx / (outW * outH * K);

    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            int ih = oh + r;  // padding=0, stride=1, dilation=1
            if (ih < 0 || ih >= H) continue;
            for (int s = 0; s < S; ++s) {
                int iw = ow + s;
                if (iw < 0 || iw >= W) continue;
                sum += input[((n * C + c) * H + ih) * W + iw] *
                       weight[((k * C + c) * R + r) * S + s];
            }
        }
    }
    sum += bias[k];
    output[idx] = sum;
}

// Launcher functions with C linkage
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad,
    cudaStream_t stream) {
    int outH = H + 2 * pad;
    int outW = W + 2 * pad;
    int total = N * C * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, outH, outW, pad);
}

extern "C" void sigmoid_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    sigmoid_kernel<<<blocks, threads, 0, stream>>>(input, output, total);
}

extern "C" void add_bias_launcher(
    const float* input,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    cudaStream_t stream) {
    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(input, bias, output, N, C, H, W);
}

extern "C" void conv2d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int outH, int outW,
    cudaStream_t stream) {
    int total = N * K * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    conv2d_naive_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C, H, W, K, R, S, outH, outW);
}
