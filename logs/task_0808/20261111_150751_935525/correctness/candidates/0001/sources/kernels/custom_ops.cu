#include <cuda_runtime.h>

// Constant padding along last two dimensions: pad[0]=left, pad[1]=right, pad[2]=top, pad[3]=bottom
__global__ void constant_pad_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * (H + pad_top + pad_bottom) * (W + pad_left + pad_right);
    if (idx >= total) return;

    int out_w = idx % (W + pad_left + pad_right);
    int out_h = (idx / (W + pad_left + pad_right)) % (H + pad_top + pad_bottom);
    int c = (idx / ((W + pad_left + pad_right) * (H + pad_top + pad_bottom))) % C;
    int n = idx / ((W + pad_left + pad_right) * (H + pad_top + pad_bottom) * C);

    if (out_h < pad_top || out_h >= pad_top + H || out_w < pad_left || out_w >= pad_left + W) {
        output[idx] = value;
    } else {
        int in_h = out_h - pad_top;
        int in_w = out_w - pad_left;
        int input_idx = ((n * C + c) * H + in_h) * W + in_w;
        output[idx] = input[input_idx];
    }
}

// Stack duplicate along dim=1: output shape [N, 2, C, H, W]
__global__ void stack_duplicate_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = 2 * N * C * H * W;
    if (idx >= total) return;

    int in_idx = idx % (N * C * H * W);
    output[idx] = input[in_idx];
}

extern "C" void constant_pad_launcher(
    float* output,
    const float* input,
    int64_t n,
    int64_t c,
    int64_t h,
    int64_t w,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom,
    float value,
    cudaStream_t stream)
{
    int out_h = h + pad_top + pad_bottom;
    int out_w = w + pad_left + pad_right;
    int64_t total = n * c * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    constant_pad_kernel<<<blocks, threads, 0, stream>>>(
        output, input,
        (int)n, (int)c, (int)h, (int)w,
        (int)pad_left, (int)pad_right, (int)pad_top, (int)pad_bottom,
        value);
}

extern "C" void stack_duplicate_launcher(
    float* output,
    const float* input,
    int64_t n,
    int64_t c,
    int64_t h,
    int64_t w,
    cudaStream_t stream)
{
    int64_t total = 2 * n * c * h * w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    stack_duplicate_kernel<<<blocks, threads, 0, stream>>>(
        output, input,
        (int)n, (int)c, (int)h, (int)w);
}
