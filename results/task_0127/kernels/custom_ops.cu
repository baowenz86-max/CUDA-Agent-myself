#include <cuda_runtime.h>

// Reflection padding kernel.
// Supports arbitrary input strides by using stride parameters.
// Output tensor is assumed contiguous with shape [N, C, H_out, W_out].
// Each thread computes one output element.
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int pad_top, int pad_bottom, int pad_left, int pad_right,
    long long input_stride_n,   // stride to move batch
    long long input_stride_c,   // stride to move channel
    long long input_stride_h,   // stride to move height
    long long input_stride_w    // stride to move width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    // Decompose linear index into n, c, oh, ow
    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;

    // Compute reflected input coordinates
    int ih = oh - pad_top;
    // reflect if out of range
    if (ih < 0) ih = -ih;
    int max_ih = H_in - 1;
    if (ih > max_ih) {
        int period = 2 * H_in - 2;
        ih = ih % period;
        if (ih >= H_in) ih = period - ih;
    }

    int iw = ow - pad_left;
    if (iw < 0) iw = -iw;
    int max_iw = W_in - 1;
    if (iw > max_iw) {
        int period = 2 * W_in - 2;
        iw = iw % period;
        if (iw >= W_in) iw = period - iw;
    }

    // Compute input offset using strides
    long long input_offset = n * input_stride_n + c * input_stride_c
                             + ih * input_stride_h + iw * input_stride_w;
    output[idx] = input[input_offset];
}

// Launcher function (C interface)
// input: pointer to input data (float)
// output: pointer to output data (float, preallocated contiguous)
// N,C,H_in,W_in: input dims
// pad_top,pad_bottom,pad_left,pad_right: padding amounts
// input_strides: array of 4 strides in elements for input tensor (n,c,h,w)
// stream: CUDA stream
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int pad_top, int pad_bottom, int pad_left, int pad_right,
    const long long* input_strides,
    cudaStream_t stream
) {
    int H_out = H_in + pad_top + pad_bottom;
    int W_out = W_in + pad_left + pad_right;
    int total = N * C * H_out * W_out;

    const int threads = 256;
    int blocks = (total + threads - 1) / threads;

    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, H_in, W_in,
        H_out, W_out,
        pad_top, pad_bottom, pad_left, pad_right,
        input_strides[0], input_strides[1], input_strides[2], input_strides[3]
    );
}
