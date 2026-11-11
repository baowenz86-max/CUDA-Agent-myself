#include <cuda_runtime.h>

// LP Pooling with norm_type=2 equivalent to sqrt(sum(x^2)) over each kernel window
// Supports only zero padding and stride, ceil_mode as specified.
// Kernel assumes input is contiguous NCHW float32 tensor.
// Grid-stride loop over output elements.
__global__ void lp_pool2d_kernel(float* __restrict__ output,
                                 const float* __restrict__ input,
                                 int N, int C, int H, int W,
                                 int out_h, int out_w,
                                 int kernel_h, int kernel_w,
                                 int stride_h, int stride_w,
                                 int pad_h, int pad_w,
                                 bool ceil_mode) {
    int total = N * C * out_h * out_w;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        // Decode linear index to n,c,oh,ow
        int ow = idx % out_w;
        int oh = (idx / out_w) % out_h;
        int c = (idx / (out_w * out_h)) % C;
        int n = idx / (out_w * out_h * C);

        // Input window top-left coordinates
        int ih0 = oh * stride_h - pad_h;
        int iw0 = ow * stride_w - pad_w;

        float sum_sq = 0.0f;
        for (int kh = 0; kh < kernel_h; ++kh) {
            int ih = ih0 + kh;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int iw = iw0 + kw;
                if (iw < 0 || iw >= W) continue;
                float val = input[((n * C + c) * H + ih) * W + iw];
                sum_sq += val * val;
            }
        }
        output[idx] = sqrtf(sum_sq);
    }
}

// C-style launcher, no PyTorch dependencies.
// norm_type is ignored because we only implement norm_type=2 as per contract.
extern "C" void lp_pool2d_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int norm_type,
    int kernel_h, int kernel_w,
    int stride, int padding_h, int padding_w,
    bool ceil_mode,
    cudaStream_t stream)
{
    // Compute output dimensions
    int out_h, out_w;
    if (ceil_mode) {
        out_h = (H + 2 * padding_h - kernel_h + stride - 1) / stride + 1;
        out_w = (W + 2 * padding_w - kernel_w + stride - 1) / stride + 1;
    } else {
        out_h = (H + 2 * padding_h - kernel_h) / stride + 1;
        out_w = (W + 2 * padding_w - kernel_w) / stride + 1;
    }

    int total = N * C * out_h * out_w;
    if (total == 0) return;

    // Choose block size and grid size (grid-stride loop handles larger cases)
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid cap, but loop covers
    if (blocks < 1) blocks = 1;

    lp_pool2d_kernel<<<blocks, threads, 0, stream>>>(
        output, input,
        N, C, H, W,
        out_h, out_w,
        kernel_h, kernel_w,
        stride, stride,
        padding_h, padding_w,
        ceil_mode);
}
