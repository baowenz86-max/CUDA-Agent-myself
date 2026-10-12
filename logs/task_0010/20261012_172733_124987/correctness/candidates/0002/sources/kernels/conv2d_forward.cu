#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Custom conv2d forward kernel with bias.
// Assumes input, weight, bias are contiguous float32 tensors.
// Input layout: NCHW, weight layout: OC,IC,KH,KW.
__global__ void conv2d_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int batch,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int out_h,
    int out_w
) {
    int total_elements = batch * out_channels * out_h * out_w;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Decompose linear index into (n, oc, oh, ow)
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int oc = (idx / (out_w * out_h)) % out_channels;
    int n = idx / (out_w * out_h * out_channels);

    float acc = bias[oc];
    int pad_h = kernel_h / 2;
    int pad_w = kernel_w / 2;

    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = oh + kh - pad_h;
            if (ih < 0 || ih >= in_h) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int iw = ow + kw - pad_w;
                if (iw < 0 || iw >= in_w) continue;
                int input_idx = ((n * in_channels + ic) * in_h + ih) * in_w + iw;
                int weight_idx = ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                acc += input[input_idx] * weight[weight_idx];
            }
        }
    }
    output[idx] = acc;
}

// C-style launcher function called from conv2d_binding.cpp
extern "C" void conv2d_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int batch,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int kernel_h,
    int kernel_w,
    cudaStream_t stream
) {
    int pad_h = kernel_h / 2;
    int pad_w = kernel_w / 2;
    int out_h = (in_h + 2 * pad_h - kernel_h) + 1;
    int out_w = (in_w + 2 * pad_w - kernel_w) + 1;
    int total_elements = batch * out_channels * out_h * out_w;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    conv2d_forward_kernel<<<blocks, threads, 0, stream>>>(
        output, input, weight, bias,
        batch, in_channels, in_h, in_w, out_channels, kernel_h, kernel_w, out_h, out_w
    );
}
