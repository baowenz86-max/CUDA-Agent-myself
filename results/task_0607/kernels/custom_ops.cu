#include <cuda_runtime.h>

// CUDA kernel for unfold + abs
// Each thread processes one output element.
__global__ void unfold_abs_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int dilation,
    int padding_h, int padding_w,
    int stride_h, int stride_w)
{
    // Compute output dimensions
    int out_h = (H + 2 * padding_h - dilation * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (W + 2 * padding_w - dilation * (kernel_w - 1) - 1) / stride_w + 1;
    int out_spatial = out_h * out_w;
    int CK = C * kernel_h * kernel_w;
    int total = N * CK * out_spatial;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        int l = idx % out_spatial;
        int tmp = idx / out_spatial;
        int ck = tmp % CK;
        int n = tmp / CK;

        // Decode ck into c, kh_idx, kw_idx
        int c = ck / (kernel_h * kernel_w);
        int khw = ck % (kernel_h * kernel_w);
        int kh = khw / kernel_w;
        int kw = khw % kernel_w;

        // Compute spatial position from l
        int h_idx = l / out_w;
        int w_idx = l % out_w;

        // Input coordinates
        int h_in = h_idx * stride_h - padding_h + kh * dilation;
        int w_in = w_idx * stride_w - padding_w + kw * dilation;

        float val = 0.0f;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            val = input[((n * C + c) * H + h_in) * W + w_in];
        }
        // abs
        output[idx] = fabsf(val);
    }
}

// Launcher function (C interface)
extern "C" void launch_custom_unfold_abs(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int dilation,
    int padding_h, int padding_w,
    int stride_h, int stride_w,
    cudaStream_t stream)
{
    int out_h = (H + 2 * padding_h - dilation * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (W + 2 * padding_w - dilation * (kernel_w - 1) - 1) / stride_w + 1;
    int out_spatial = out_h * out_w;
    int CK = C * kernel_h * kernel_w;
    int total = N * CK * out_spatial;

    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid cap; kernel uses grid-stride loop

    unfold_abs_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, H, W,
        kernel_h, kernel_w,
        dilation,
        padding_h, padding_w,
        stride_h, stride_w);
}
