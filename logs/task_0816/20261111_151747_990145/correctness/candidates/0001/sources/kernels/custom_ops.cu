#include <cuda_runtime.h>

// CUDA kernel for bilinear upsampling, align_corners=false
// Input: x (float*) shape [N, C, H, W]
// Output: y (float*) shape [N, C, H_out, W_out]
// scale_factor: multiplier for H and W
__global__ void upsample_bilinear2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int H_out, int W_out,
    float scale_factor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (W_out * H_out * C);

    // Compute source coordinates (align_corners=false)
    float src_h = (h_out + 0.5f) / scale_factor - 0.5f;
    float src_w = (w_out + 0.5f) / scale_factor - 0.5f;

    // Clamp to valid range
    if (src_h < 0) src_h = 0;
    if (src_w < 0) src_w = 0;
    if (src_h > H - 1) src_h = H - 1;
    if (src_w > W - 1) src_w = W - 1;

    int h0 = (int)src_h;
    int w0 = (int)src_w;
    int h1 = (h0 < H - 1) ? h0 + 1 : h0;
    int w1 = (w0 < W - 1) ? w0 + 1 : w0;

    float h_lambda = src_h - h0;
    float w_lambda = src_w - w0;

    const float* base = x + ((long long)n * C + c) * H * W;
    float v00 = base[h0 * W + w0];
    float v01 = base[h0 * W + w1];
    float v10 = base[h1 * W + w0];
    float v11 = base[h1 * W + w1];

    float top = v00 * (1.0f - w_lambda) + v01 * w_lambda;
    float bottom = v10 * (1.0f - w_lambda) + v11 * w_lambda;
    float value = top * (1.0f - h_lambda) + bottom * h_lambda;

    y[idx] = value;
}

// CUDA kernel for mean along dim=1 keepdim
// Input: x (float*) shape [N, C, H, W]
// Output: y (float*) shape [N, 1, H, W]
__global__ void mean_dim1_keepdim_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int n = idx / (W * H);

    float sum = 0.0f;
    const float* base = x + ((long long)n * C) * H * W;
    for (int c = 0; c < C; ++c) {
        sum += base[(c * H + h) * W + w];
    }
    y[idx] = sum / C;
}

// Launcher for custom_upsample_bilinear2d
// x: input pointer (CUDA float*)
// y: output pointer (CUDA float*)
// N,C,H,W: input dims
// scale_factor: float
// align_corners: bool (only false supported)
// stream: CUDA stream
extern "C" void custom_upsample_bilinear2d_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    float scale_factor,
    bool align_corners,
    cudaStream_t stream)
{
    // align_corners currently unused; only false case is needed
    int H_out = (int)(H * scale_factor);
    int W_out = (int)(W * scale_factor);
    int total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    upsample_bilinear2d_kernel<<<blocks, threads, 0, stream>>>(
        x, y, N, C, H, W, H_out, W_out, scale_factor);
}

// Launcher for custom_mean_dim1_keepdim
// x: input pointer (CUDA float*)
// y: output pointer (CUDA float*)
// N,C,H,W: input dims
// stream: CUDA stream
extern "C" void custom_mean_dim1_keepdim_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int total = N * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mean_dim1_keepdim_kernel<<<blocks, threads, 0, stream>>>(
        x, y, N, C, H, W);
}
