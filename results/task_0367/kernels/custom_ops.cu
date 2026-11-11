#include <cuda_runtime.h>

// CUDA kernel for adaptive average pooling 2D
// Each thread computes one output element
__global__ void adaptive_avg_pool2d_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int N, int C, int H, int W,
                                           int out_h, int out_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * out_h * out_w;
    if (idx >= total) return;

    // Decode output indices: n, c, oh, ow
    int ow = idx % out_w;
    int tmp = idx / out_w;
    int oh = tmp % out_h;
    tmp = tmp / out_h;
    int c = tmp % C;
    int n = tmp / C;

    // Compute pooling region boundaries for this output position
    int start_h = (int)floorf((float)oh * H / out_h);
    int end_h   = (int)ceilf((float)(oh + 1) * H / out_h);
    int start_w = (int)floorf((float)ow * W / out_w);
    int end_w   = (int)ceilf((float)(ow + 1) * W / out_w);

    // Clamp to valid range (should be within bounds by formula, but safe)
    if (start_h < 0) start_h = 0;
    if (end_h > H) end_h = H;
    if (start_w < 0) start_w = 0;
    if (end_w > W) end_w = W;

    float sum = 0.0f;
    int count = 0;

    // Base pointer for this (n, c) slice
    const float* slice = input + ((long long)n * C + c) * H * W; // long long for large sizes

    for (int h = start_h; h < end_h; ++h) {
        for (int w = start_w; w < end_w; ++w) {
            sum += slice[h * W + w];
            ++count;
        }
    }

    // Output index: ((n * C + c) * out_h + oh) * out_w + ow
    int out_idx = ((n * C + c) * out_h + oh) * out_w + ow;
    output[out_idx] = sum / (float)count;
}

// Launcher function with C linkage for binding
extern "C" void custom_adaptive_avg_pool2d_forward_launcher(const float* input,
                                                               float* output,
                                                               int N, int C, int H, int W,
                                                               int out_h, int out_w,
                                                               cudaStream_t stream) {
    int total = N * C * out_h * out_w;
    if (total <= 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    adaptive_avg_pool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, out_h, out_w
    );
}
