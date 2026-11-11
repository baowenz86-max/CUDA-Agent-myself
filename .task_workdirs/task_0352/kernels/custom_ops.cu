#include <cuda_runtime.h>
#include <cstdint>

// Bilinear upsample kernel: align_corners=false
__global__ void upsample_bilinear2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out, float scale_h, float scale_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int nc = idx / (W_out * H_out);

    // Source coordinates, align_corners=false:
    // src = (dst + 0.5) / scale - 0.5
    float h_in_f = (h_out + 0.5f) * scale_h - 0.5f;
    float w_in_f = (w_out + 0.5f) * scale_w - 0.5f;

    // Clamp to [-0.5, H_in-0.5] (PyTorch behavior with edge padding)
    h_in_f = fmaxf(0.0f, fminf(h_in_f, H_in - 1.0f));
    w_in_f = fmaxf(0.0f, fminf(w_in_f, W_in - 1.0f));

    int h0 = (int)h_in_f;
    int w0 = (int)w_in_f;
    int h1 = min(h0 + 1, H_in - 1);
    int w1 = min(w0 + 1, W_in - 1);

    float lambda_h = h_in_f - h0;
    float lambda_w = w_in_f - w0;

    const float* base = input + nc * H_in * W_in;
    float val = 
        (1.0f - lambda_h) * (1.0f - lambda_w) * base[h0 * W_in + w0] +
        (1.0f - lambda_h) * lambda_w * base[h0 * W_in + w1] +
        lambda_h * (1.0f - lambda_w) * base[h1 * W_in + w0] +
        lambda_h * lambda_w * base[h1 * W_in + w1];

    output[idx] = val;
}

// Reflection padding kernel: padding (left, right, top, bottom) = (1,1,1,1)
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int pad_left, int pad_right, int pad_top, int pad_bottom)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H_in + pad_top + pad_bottom;
    int W_out = W_in + pad_left + pad_right;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int nc = idx / (W_out * H_out);

    // Map to source index using reflection
    int h_in = h_out - pad_top;
    if (h_in < 0) h_in = -h_in;
    if (h_in >= H_in) h_in = 2 * H_in - 2 - h_in;

    int w_in = w_out - pad_left;
    if (w_in < 0) w_in = -w_in;
    if (w_in >= W_in) w_in = 2 * W_in - 2 - w_in;

    output[idx] = input[nc * H_in * W_in + h_in * W_in + w_in];
}

extern "C" void upsample_bilinear2d_launcher(
    const float* input, float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out, double scale_factor,
    cudaStream_t stream)
{
    float scale_h = (float)(1.0 / scale_factor); // scale = 1/scale_factor because output size = floor(input*scale_factor)
    float scale_w = scale_h;

    int total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    upsample_bilinear2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out, scale_h, scale_w);
}

extern "C" void reflection_pad2d_launcher(
    const float* input, float* output,
    int N, int C, int H_in, int W_in,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    cudaStream_t stream)
{
    int H_out = H_in + pad_top + pad_bottom;
    int W_out = W_in + pad_left + pad_right;
    int total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H_in, W_in,
        pad_left, pad_right, pad_top, pad_bottom);
}
