#include <cuda_runtime.h>

// 2D reflection padding kernel: supports arbitrary padding on last two dims.
__global__ void reflection_pad2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int pad_left,
    int pad_top
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * out_h * out_w;
    if (idx >= total) return;

    // Decode linear index to output coordinates: (b, c, oh, ow)
    int ow = idx % out_w;
    int tmp = idx / out_w;
    int oh = tmp % out_h;
    tmp /= out_h;
    int c = tmp % channels;
    int b = tmp / channels;

    // Map output position to input position with reflection
    int iw = ow - pad_left;
    if (iw < 0) iw = -iw;
    if (iw >= in_w) iw = 2 * (in_w - 1) - iw;

    int ih = oh - pad_top;
    if (ih < 0) ih = -ih;
    if (ih >= in_h) ih = 2 * (in_h - 1) - ih;

    // Input is assumed contiguous, stride simple
    int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
    output[idx] = input[input_idx];
}

// Launcher for reflection pad 2d. Assumes input is contiguous float32 CUDA tensor.
extern "C" void reflection_pad2d_forward_launcher(
    float* output,
    const float* input,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int pad_left,
    int pad_top,
    cudaStream_t stream
) {
    int total = batch * channels * out_h * out_w;
    if (total == 0) return;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        output, input, batch, channels, in_h, in_w, out_h, out_w, pad_left, pad_top);
}
