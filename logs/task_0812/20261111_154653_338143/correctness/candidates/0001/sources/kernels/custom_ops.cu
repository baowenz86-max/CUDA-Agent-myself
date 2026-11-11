#include <cuda_runtime.h>
#include <cstdint>

// Bilinear 2x upsampling (align_corners=False)
// Input: float32 tensor shape [N, C, H, W], contiguous
// Output: float32 tensor shape [N, C, 2*H, 2*W], contiguous
// Source coordinate: (out+0.5)/scale - 0.5
template <int scale>
__global__ void upsample_bilinear2x_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (out_idx >= total) return;

    int ow = out_idx % outW;
    int tmp1 = out_idx / outW;
    int oh = tmp1 % outH;
    int nc = tmp1 / outH;

    float y = (oh + 0.5f) / scale - 0.5f;
    float x = (ow + 0.5f) / scale - 0.5f;

    y = fmaxf(0.0f, y);
    x = fmaxf(0.0f, x);
    y = fminf(y, (float)(H - 1));
    x = fminf(x, (float)(W - 1));

    int y0 = (int)y;
    int x0 = (int)x;
    int y1 = min(y0 + 1, H - 1);
    int x1 = min(x0 + 1, W - 1);

    float wy1 = y - y0;
    float wx1 = x - x0;
    float wy0 = 1.0f - wy1;
    float wx0 = 1.0f - wx1;

    const float* in_base = input + nc * H * W;
    float v00 = in_base[y0 * W + x0];
    float v01 = in_base[y0 * W + x1];
    float v10 = in_base[y1 * W + x0];
    float v11 = in_base[y1 * W + x1];

    float top = v00 * wx0 + v01 * wx1;
    float bottom = v10 * wx0 + v11 * wx1;
    output[out_idx] = top * wy0 + bottom * wy1;
}

// Rotation by 90 degrees in the last two dimensions, k mod 4.
// For k=1: output[b,c,j,H-1-i] = input[b,c,i,j]
// General modulo 4 handling.
__global__ void rot90_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int k_mod)
{
    // output shape is [N, C, W, H] for k=1/3, same as input for k=0/2
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * W * H; // same number of elements
    if (idx >= total) return;

    // output coordinates: decompose into (n, c, oj, oi) where OW=W, OH=H? Actually output shape [N,C,W,H] for k=1
    // For simplicity, use output linear index assuming output shape [N,C,W,H] for k=1/3, and [N,C,H,W] for k=0/2.
    // We will dispatch based on k_mod to avoid complex generic indexing.
    // This kernel only handles k_mod==1; other k handled by separate kernels/launch logic if needed.
}

// Specialized kernel for k_mod == 1 (most common for this model)
__global__ void rot90_k1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    // output shape [N, C, W, H]
    int outW = H;
    int outH = W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (idx >= total) return;

    int oi = idx % outW; // j index in output, corresponds to input j
    int tmp1 = idx / outW;
    int oj = tmp1 % outH; // i index in output, corresponds to input H-1-i
    int nc = tmp1 / outH;

    int input_i = H - 1 - oj;
    int input_j = oi;

    output[idx] = input[nc * H * W + input_i * W + input_j];
}

// Specialized kernel for k_mod == 3 (optional, not used in current model but implemented for completeness)
__global__ void rot90_k3_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    int outW = H;
    int outH = W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outH * outW;
    if (idx >= total) return;

    int oi = idx % outW;
    int tmp1 = idx / outW;
    int oj = tmp1 % outH;
    int nc = tmp1 / outH;

    int input_i = oj;
    int input_j = W - 1 - oi;
    output[idx] = input[nc * H * W + input_i * W + input_j];
}

// Generic rotation for k_mod=0 (copy) and k_mod=2 (180-degree rotation)
__global__ void rot90_k0_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;
    output[idx] = input[idx];
}

__global__ void rot90_k2_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;
    int w = idx % W;
    int tmp1 = idx / W;
    int h = tmp1 % H;
    int nc = tmp1 / H;
    int in_i = H - 1 - h;
    int in_j = W - 1 - w;
    output[idx] = input[nc * H * W + in_i * W + in_j];
}

extern "C" void custom_upsample_bilinear2x_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int outH = H * 2;
    int outW = W * 2;
    int total = N * C * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    upsample_bilinear2x_kernel<2><<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, outH, outW);
}

extern "C" void custom_rot90_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int k,
    cudaStream_t stream)
{
    int k_mod = ((k % 4) + 4) % 4;
    int threads = 256;
    int blocks;

    if (k_mod == 1) {
        int total = N * C * W * H; // output shape [N,C,W,H]
        blocks = (total + threads - 1) / threads;
        rot90_k1_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W);
    } else if (k_mod == 3) {
        int total = N * C * W * H;
        blocks = (total + threads - 1) / threads;
        rot90_k3_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W);
    } else if (k_mod == 2) {
        int total = N * C * H * W;
        blocks = (total + threads - 1) / threads;
        rot90_k2_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W);
    } else {
        int total = N * C * H * W;
        blocks = (total + threads - 1) / threads;
        rot90_k0_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W);
    }
}