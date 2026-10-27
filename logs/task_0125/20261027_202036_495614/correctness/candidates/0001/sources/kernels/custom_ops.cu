#include <cuda_runtime.h>

// This file contains only non-convolution CUDA kernels.
// ConvTranspose1d must be implemented with cuDNN in the C++ binding layer.
// unsqueeze and squeeze are view operations handled directly in binding.cpp.

// Elementwise BatchNorm2d (evaluation mode) kernel.
// Input layout: [N, C, 1, L] with arbitrary strides.
// Output has the same shape and strides as input when allocated with empty_like,
// but the binding layer may return a contiguous output; we write using output strides.
// Since this kernel is only called for the specific unsqueezed tensor with
// stride [C*1*L, 1*L, L, 1], we could optimize for that, but handle general strides.
__global__ void batchnorm2d_eval_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float* __restrict__ output,
    int N, int C, int L,
    float eps,
    // strides for input/output (identical as output is a copy with same shape)
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    int total = N * C * L; // total number of elements, ignoring the dummy dimension of size 1
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        // Convert flat index to n,c,l (since h dimension is always 1)
        int l = i % L;
        int tmp = i / L;
        int c = tmp % C;
        int n = tmp / C;

        int h = 0; // dummy dimension

        // Input/output linear offsets using provided strides
        long long in_offset = n * in_stride_n + c * in_stride_c + h * in_stride_h + l * in_stride_w;
        long long out_offset = n * out_stride_n + c * out_stride_c + h * out_stride_h + l * out_stride_w;

        float x = input[in_offset];
        float scale = weight[c] / sqrtf(running_var[c] + eps);
        float shift = bias[c] - running_mean[c] * scale;
        output[out_offset] = x * scale + shift;
    }
}

// Launcher for batch_norm2d in eval mode.
// All pointers must be CUDA device pointers; no PyTorch dependencies.
extern "C" void batchnorm2d_eval_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int N, int C, int L,
    float eps,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    int total = N * C * L;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // cap blocks to avoid excessive grid size; grid-stride loop handles the rest
    if (blocks > 65535) blocks = 65535;
    batchnorm2d_eval_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, running_mean, running_var, output,
        N, C, L, eps,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}

// Note: conv_transpose1d_forward must be implemented in kernels/*_binding.cpp using cuDNN.
// This .cu file must not contain conv_transpose1d kernels.
