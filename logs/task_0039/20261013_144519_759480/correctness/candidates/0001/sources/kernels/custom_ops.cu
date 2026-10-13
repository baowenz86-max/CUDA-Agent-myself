// CUDA kernels for non-convolutional custom ops: avg_pool2d and sigmoid.
// All functions are extern "C" launchers with raw pointers and cudaStream_t.
// No torch/ATen/c10 dependencies.

#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// avg_pool2d kernel
// Input: (const float* input, int N, int C, int H, int W,
//         int input_n_stride, int input_c_stride, int input_h_stride, int input_w_stride,
//         int kernel_size, int stride, int padding,
//         float* output, int outH, int outW)
// Supports arbitrary strides for input, output is contiguous (as allocated by binding)
// This kernel assumes no output shape calculation; outH and outW are passed from launcher/binding.
// Padding is symmetric and only zero-padding is supported.
__global__ void avg_pool2d_kernel(
    const float* __restrict__ input,
    int N, int C, int H, int W,
    int input_n_stride, int input_c_stride, int input_h_stride, int input_w_stride,
    int kernel_size, int stride, int padding,
    float* __restrict__ output, int outH, int outW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (idx >= total) return;

    // Decompose linear index into n, c, oh, ow
    int ow = idx % outW;
    int oh = (idx / outW) % outH;
    int c = (idx / (outW * outH)) % C;
    int n = idx / (outW * outH * C);

    float sum = 0.0f;
    int count = 0;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int ih = oh * stride - padding + kh;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int iw = ow * stride - padding + kw;
            if (iw < 0 || iw >= W) continue;
            // Input element offset using strides
            const float* in_ptr = input + n * input_n_stride + c * input_c_stride + ih * input_h_stride + iw * input_w_stride;
            sum += *in_ptr;
            count++;
        }
    }
    // Average over all kernel positions (including zero paddings) as per AvgPool2d semantics
    float avg = sum / (float)(kernel_size * kernel_size);
    output[idx] = avg;
}

extern "C" void avg_pool2d_launcher(
    const float* input,
    int N, int C, int H, int W,
    int input_n_stride, int input_c_stride, int input_h_stride, int input_w_stride,
    int kernel_size, int stride, int padding,
    float* output, int outH, int outW,
    cudaStream_t stream)
{
    int total = N * C * outH * outW;
    if (total <= 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    avg_pool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, N, C, H, W,
        input_n_stride, input_c_stride, input_h_stride, input_w_stride,
        kernel_size, stride, padding,
        output, outH, outW);
}

// -----------------------------------------------------------------------------
// sigmoid kernel: element-wise sigmoid on arbitrary input layout, output contiguous
// Input and output are 1D arrays of the same number of elements.
__global__ void sigmoid_kernel(const float* __restrict__ input, float* __restrict__ output, int numel)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float x = input[idx];
        // Stable sigmoid: 1 / (1 + exp(-x))
        output[idx] = 1.0f / (1.0f + expf(-x));
    }
}

extern "C" void sigmoid_launcher(
    const float* input,
    float* output,
    int numel,
    cudaStream_t stream)
{
    if (numel <= 0) return;

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}
