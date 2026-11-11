// kernels/custom_ops.cu
// CUDA implementation of unfold (im2col) matching torch.nn.functional.unfold with stride=1, padding=0.
// Only float32 input/output. Input may be non-contiguous; launcher receives strides.

#include <cuda_runtime.h>

// Kernel: copy elements from input to output according to unfold layout.
// output shape: [N, C*K, L] where K = kernel_h * kernel_w, L = out_h * out_w.
// Global index idx in output maps to (n, c*K + k, l).
// For each thread, we use grid-stride loop to cover all elements.
__global__ void unfold_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int dilation_h, int dilation_w,
    int out_h, int out_w,
    int inp_sN, int inp_sC, int inp_sH, int inp_sW)
{
    int K = kernel_h * kernel_w;
    int L = out_h * out_w;
    int total = N * C * K * L;
    int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        int n = idx / (C * K * L);
        int rem = idx % (C * K * L);
        int c = rem / (K * L);
        int rem2 = rem % (K * L);
        int k = rem2 / L;
        int l = rem2 % L;
        int kh = k / kernel_w;
        int kw = k % kernel_w;
        int oh = l / out_w;
        int ow = l % out_w;
        int ih = oh + kh * dilation_h;
        int iw = ow + kw * dilation_w;
        // Input offset: n*inp_sN + c*inp_sC + ih*inp_sH + iw*inp_sW
        int in_idx = n * inp_sN + c * inp_sC + ih * inp_sH + iw * inp_sW;
        output[idx] = input[in_idx];
    }
}

// Host launcher: no PyTorch dependencies. Takes preallocated contiguous output.
// Strides for input are passed separately to support non-contiguous tensors.
extern "C" void unfold_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int dilation_h, int dilation_w,
    int inp_sN, int inp_sC, int inp_sH, int inp_sW,
    cudaStream_t stream)
{
    int out_h = H - dilation_h * (kernel_h - 1);
    int out_w = W - dilation_w * (kernel_w - 1);
    int total = N * C * kernel_h * kernel_w * out_h * out_w;
    if (total <= 0) return;

    const int block = 256;
    // Choose grid size to cover all elements, but cap to avoid excessive blocks.
    // Grid-stride loop handles overflow.
    int grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535;
    unfold_kernel<<<grid, block, 0, stream>>>(
        input, output,
        N, C, H, W,
        kernel_h, kernel_w,
        dilation_h, dilation_w,
        out_h, out_w,
        inp_sN, inp_sC, inp_sH, inp_sW);
}
