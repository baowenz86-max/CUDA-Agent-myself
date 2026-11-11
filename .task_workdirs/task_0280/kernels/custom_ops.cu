#include <cuda_runtime.h>
#include <math.h>

// ReflectionPad2d kernel: outputs are contiguous, input may have arbitrary strides.
// padding is assumed symmetric (same for all sides) as per contract.
__global__ void reflection_pad2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int padding,
    int64_t in_n_stride, int64_t in_c_stride, int64_t in_h_stride, int64_t in_w_stride)
{
    long long total = (long long)N * C * H_out * W_out;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int w_out = idx % W_out;
        long long tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        // compute source coordinates with reflection
        int h_in = h_out - padding;
        int w_in = w_out - padding;

        // reflect if out of bounds
        h_in = (h_in < 0) ? -h_in : h_in;
        if (h_in >= H_in) h_in = 2 * (H_in - 1) - h_in;
        w_in = (w_in < 0) ? -w_in : w_in;
        if (w_in >= W_in) w_in = 2 * (W_in - 1) - w_in;

        // input may not be contiguous; use strides
        const float* src = input + (long long)n * in_n_stride + (long long)c * in_c_stride
                           + (long long)h_in * in_h_stride + (long long)w_in * in_w_stride;
        output[idx] = *src;
    }
}

// Log2 kernel: elementwise log2f, handles arbitrary strides, output contiguous.
__global__ void log2_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long numel,
    int64_t in_n_stride, int64_t in_c_stride, int64_t in_h_stride, int64_t in_w_stride,
    int N, int C, int H, int W)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < numel; idx += stride) {
        // decompose idx into n,c,h,w
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;

        const float* src = input + (long long)n * in_n_stride + (long long)c * in_c_stride
                           + (long long)h * in_h_stride + (long long)w * in_w_stride;
        output[idx] = log2f(*src);
    }
}

// Launcher for reflection_pad2d_forward.
// Input: x (float32 CUDA tensor, arbitrary strides), padding (int).
// Output: newly allocated contiguous float32 tensor of shape [N, C, H+2p, W+2p].
// The allocation is done by the binding layer; this launcher receives the data pointer.
extern "C" void reflection_pad2d_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H_in, int W_in,
    int padding,
    int64_t in_n_stride, int64_t in_c_stride, int64_t in_h_stride, int64_t in_w_stride,
    cudaStream_t stream)
{
    int H_out = H_in + 2 * padding;
    int W_out = W_in + 2 * padding;
    long long total = (long long)N * C * H_out * W_out;
    if (total == 0) return;

    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;  // cap grid to avoid launch failure; kernel uses grid-stride loop

    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        output, input, N, C, H_in, W_in, H_out, W_out, padding,
        in_n_stride, in_c_stride, in_h_stride, in_w_stride);
}

// Launcher for log2_forward.
// Input: x (float32 CUDA tensor, arbitrary strides).
// Output: newly allocated contiguous float32 tensor of same shape.
extern "C" void log2_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int64_t in_n_stride, int64_t in_c_stride, int64_t in_h_stride, int64_t in_w_stride,
    cudaStream_t stream)
{
    long long numel = (long long)N * C * H * W;
    if (numel == 0) return;

    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    log2_kernel<<<blocks, threads, 0, stream>>>(
        output, input, numel,
        in_n_stride, in_c_stride, in_h_stride, in_w_stride,
        N, C, H, W);
}
