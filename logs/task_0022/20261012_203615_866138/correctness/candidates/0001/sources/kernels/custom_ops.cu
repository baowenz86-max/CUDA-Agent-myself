// custom_ops.cu
#include <cuda_runtime.h>

// Kernel: compute L2 norm over channel dimension (dim=1) with keepdim=True.
// Input: [N, C, H, W] float32
// Output: [N, 1, H, W] float32
// Each thread handles one spatial location (n, h, w) and reduces over channels.
__global__ void norm2_channel_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int in_stride_n, int in_stride_c, int in_stride_h, int in_stride_w,
    int out_stride_n, int out_stride_c, int out_stride_h, int out_stride_w)
{
    int total = N * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        float sum_sq = 0.0f;
        const float* in_base = input + n * in_stride_n + h * in_stride_h + w * in_stride_w;
        for (int c = 0; c < C; ++c) {
            float val = in_base[c * in_stride_c];
            sum_sq += val * val;
        }
        float norm = sqrtf(sum_sq);
        output[n * out_stride_n + h * out_stride_h + w * out_stride_w] = norm;
    }
}

// Launcher for norm2_channel.
extern "C" void norm2_channel_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int in_stride_n, int in_stride_c, int in_stride_h, int in_stride_w,
    int out_stride_n, int out_stride_c, int out_stride_h, int out_stride_w,
    cudaStream_t stream)
{
    int total = N * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Cap blocks to avoid excessive grid dimension; grid-stride loop handles remainder.
    if (blocks > 65535) blocks = 65535;
    norm2_channel_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}

// Kernel: bilinear upsampling (2D) for float32.
// Input: [N, C, Hin, Win] float32
// Output: [N, C, Hout, Wout] float32
// Supports align_corners true/false.
// Each thread computes one output element.
__global__ void upsample_bilinear2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int Hin, int Win,
    int Hout, int Wout,
    int in_stride_n, int in_stride_c, int in_stride_h, int in_stride_w,
    int out_stride_n, int out_stride_c, int out_stride_h, int out_stride_w,
    bool align_corners)
{
    int total = N * C * Hout * Wout;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int w_out = i % Wout;
        int tmp = i / Wout;
        int h_out = tmp % Hout;
        tmp /= Hout;
        int c = tmp % C;
        int n = tmp / C;

        float h_in, w_in;
        if (align_corners) {
            // PyTorch formula when align_corners=True
            h_in = (Hout > 1) ? (float)h_out * (Hin - 1) / (Hout - 1) : 0.0f;
            w_in = (Wout > 1) ? (float)w_out * (Win - 1) / (Wout - 1) : 0.0f;
        } else {
            // PyTorch formula when align_corners=False
            h_in = ((float)h_out + 0.5f) * (Hin / (float)Hout) - 0.5f;
            w_in = ((float)w_out + 0.5f) * (Win / (float)Wout) - 0.5f;
            h_in = maxf(0.0f, h_in);
            w_in = maxf(0.0f, w_in);
        }

        int h0 = (int)h_in;
        int w0 = (int)w_in;
        int h1 = h0 + 1;
        int w1 = w0 + 1;

        // Clamp indices to valid range for edge cases (align_corners=False may produce -0.5 or slightly out)
        h0 = max(0, min(h0, Hin - 1));
        h1 = max(0, min(h1, Hin - 1));
        w0 = max(0, min(w0, Win - 1));
        w1 = max(0, min(w1, Win - 1));

        float h_lambda = h_in - (float)h0;
        float w_lambda = w_in - (float)w0;

        const float* in_n_c = input + n * in_stride_n + c * in_stride_c;
        float v00 = in_n_c[h0 * in_stride_h + w0 * in_stride_w];
        float v01 = in_n_c[h0 * in_stride_h + w1 * in_stride_w];
        float v10 = in_n_c[h1 * in_stride_h + w0 * in_stride_w];
        float v11 = in_n_c[h1 * in_stride_h + w1 * in_stride_w];

        float top = v00 * (1.0f - w_lambda) + v01 * w_lambda;
        float bottom = v10 * (1.0f - w_lambda) + v11 * w_lambda;
        float result = top * (1.0f - h_lambda) + bottom * h_lambda;

        output[n * out_stride_n + c * out_stride_c + h_out * out_stride_h + w_out * out_stride_w] = result;
    }
}

// Launcher for bilinear upsampling.
extern "C" void upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int N, int C, int Hin, int Win,
    int Hout, int Wout,
    int in_stride_n, int in_stride_c, int in_stride_h, int in_stride_w,
    int out_stride_n, int out_stride_c, int out_stride_h, int out_stride_w,
    bool align_corners,
    cudaStream_t stream)
{
    int total = N * C * Hout * Wout;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    upsample_bilinear2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, Hin, Win, Hout, Wout,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        align_corners);
}
