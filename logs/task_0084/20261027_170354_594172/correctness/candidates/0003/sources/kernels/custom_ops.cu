#include <cuda_runtime.h>
#include <cstdio>

// Non-convolution CUDA kernels for the model.
// Conv2d is handled by cuDNN in a separate .cpp binding file.
// This file contains only CUDA kernels and launchers using raw pointers/scalars/cudaStream_t.

// Kernel for fused bias add + ELU: y = ELU(x + bias)
// Input: 4D tensor with shape [N, C, H, W], possibly non-contiguous.
// Bias: shape [C, 1, 1], broadcast along H and W.
// Output: same shape as input.
__global__ void elu_add_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ bias,
    int64_t total_elements,
    int C,
    int H,
    int W,
    int input_stride_n,
    int input_stride_c,
    int input_stride_h,
    int output_stride_n,
    int output_stride_c,
    int output_stride_h,
    float alpha
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        // Compute n, c, h, w from linear index
        int w = i % W;
        int64_t tmp = i / W;
        int h = tmp % H;
        tmp = tmp / H;
        int c = tmp % C;
        int n = tmp / C;

        int64_t input_idx = (int64_t)n * input_stride_n +
                            (int64_t)c * input_stride_c +
                            (int64_t)h * input_stride_h +
                            w;  // w stride assumed 1 for contiguous last dim (but we use general strides)
        // Actually we should use input_stride_w as well, but for typical tensors it's 1.
        // To be fully general, we need input_stride_w. We'll add it.
        // Note: we don't have input_stride_w in the signature, so modify.
        // We'll rewrite kernel with full strides.
    }
}

// Corrected kernel with all strides.
__global__ void elu_add_kernel_full(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ bias,
    int64_t total_elements,
    int C,
    int H,
    int W,
    int input_stride_n,
    int input_stride_c,
    int input_stride_h,
    int input_stride_w,
    int output_stride_n,
    int output_stride_c,
    int output_stride_h,
    int output_stride_w,
    float alpha
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        int w = (int)(i % W);
        int64_t tmp = i / W;
        int h = (int)(tmp % H);
        tmp = tmp / H;
        int c = (int)(tmp % C);
        int n = (int)(tmp / C);

        int64_t input_idx = (int64_t)n * input_stride_n +
                            (int64_t)c * input_stride_c +
                            (int64_t)h * input_stride_h +
                            (int64_t)w * input_stride_w;
        int64_t output_idx = (int64_t)n * output_stride_n +
                             (int64_t)c * output_stride_c +
                             (int64_t)h * output_stride_h +
                             (int64_t)w * output_stride_w;

        float val = input[input_idx] + bias[c];
        // ELU: alpha * (exp(val) - 1) if val < 0 else val
        output[output_idx] = (val > 0.0f) ? val : alpha * (expf(val) - 1.0f);
    }
}

// Launcher for fused bias add + ELU.
// All tensors are float32 CUDA tensors. The function takes raw pointers and strides.
// It is callable from .cpp binding code.
extern "C" void elu_add_forward_launcher(
    float* output,
    const float* input,
    const float* bias,
    int64_t total_elements,
    int C,
    int H,
    int W,
    int input_stride_n,
    int input_stride_c,
    int input_stride_h,
    int input_stride_w,
    int output_stride_n,
    int output_stride_c,
    int output_stride_h,
    int output_stride_w,
    float alpha,
    cudaStream_t stream
) {
    if (total_elements <= 0) return;

    const int block_size = 256;
    int64_t grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 2147483647) {
        grid_size = 2147483647; // max grid dim
    }

    elu_add_kernel_full<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        output, input, bias,
        total_elements,
        C, H, W,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        output_stride_n, output_stride_c, output_stride_h, output_stride_w,
        alpha
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a binding, we cannot throw, but we can print error and maybe abort.
        // Since this is raw launcher, we'll just return; the caller should check cudaGetLastError.
        // But we can also use printf for debugging. We'll leave a comment.
        printf("elu_add_forward_launcher: kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

// Note: Conv2d forward is implemented using cuDNN in a separate .cpp file.
// No CUDA kernel for conv here.
