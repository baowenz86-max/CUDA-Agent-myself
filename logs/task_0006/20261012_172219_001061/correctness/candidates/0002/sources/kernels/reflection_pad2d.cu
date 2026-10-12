#include <cuda_runtime.h>

__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int pad_left,
    int pad_right,
    int pad_top,
    int pad_bottom,
    long total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (long linear_idx = idx; linear_idx < total_elements; linear_idx += stride) {
        int ow = linear_idx % out_w;
        int oh = (linear_idx / out_w) % out_h;
        long plane = linear_idx / (long)(out_w * out_h);
        int c = plane % channels;
        int b = plane / channels;

        int iw = ow - pad_left;
        int ih = oh - pad_top;

        // reflect indices
        if (iw < 0) iw = -iw;
        if (iw >= in_w) iw = 2 * in_w - 2 - iw;
        if (ih < 0) ih = -ih;
        if (ih >= in_h) ih = 2 * in_h - 2 - ih;

        long input_idx = ((long)b * channels + c) * in_h * in_w + (long)ih * in_w + iw;
        output[linear_idx] = input[input_idx];
    }
}

extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int in_h,
    int in_w,
    int out_h,
    int out_w,
    int pad_left,
    int pad_right,
    int pad_top,
    int pad_bottom,
    long total_elements,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch, channels, in_h, in_w, out_h, out_w,
        pad_left, pad_right, pad_top, pad_bottom, total_elements);
}
