#include <cuda_runtime.h>

// ReflectionPad2d kernel for float32, padding=1 on all sides.
// Input: (N, C, H, W) contiguous, output: (N, C, H+2, W+2).
// Reflection padding without repeating edge pixel: for output index i in 0..H+1,
// source index = reflect(i - 1, H) where reflect(x, size) = abs(x) if x < size else 2*size-2-x (clamped).
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
        // Reflect index
        if (iw < 0) iw = -iw;
        if (iw >= W) iw = 2 * W - 2 - iw;
        if (ih < 0) ih = -ih;
        if (ih >= H) ih = 2 * H - 2 - ih;
        // Clamp to valid range
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

// Launcher functions with C linkage, to be called from binding.cpp.
// All take raw pointers and cudaStream_t.

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
    // Limit blocks to a reasonable maximum (grid-stride loop handles rest)
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
