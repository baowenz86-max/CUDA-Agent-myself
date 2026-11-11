#include <cuda_runtime.h>

// Bilinear upsampling kernel (align_corners=False)
// Input: float32 tensor of shape [N, C, H, W], contiguous
// Output: float32 tensor of shape [N, C, outH, outW]
// scale_factor: float, output size = floor(input_size * scale_factor)
// For each output element, compute source coordinates and interpolate.
__global__ void upsample_bilinear_2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    float scale_factor)
{
    int total = N * C * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // grid-stride loop to ensure all elements covered
    for (int i = idx; i < total; i += stride) {
        // Decompose linear index into n, c, oh, ow
        int ow = i % outW;
        int tmp = i / outW;
        int oh = tmp % outH;
        tmp /= outH;
        int c = tmp % C;
        int n = tmp / C;

        // Source coordinate mapping: src = (dst + 0.5) / scale - 0.5
        float src_y = (oh + 0.5f) / scale_factor - 0.5f;
        float src_x = (ow + 0.5f) / scale_factor - 0.5f;

        // Do NOT clamp src before floor/ceil; compute integer indices and clamp them separately
        int y0 = (int)floorf(src_y);
        int x0 = (int)floorf(src_x);
        int y1 = y0 + 1;
        int x1 = x0 + 1;

        // Clamp indices to [0, H-1] and [0, W-1]
        if (y0 < 0) y0 = 0;
        if (y0 > H - 1) y0 = H - 1;
        if (y1 < 0) y1 = 0;
        if (y1 > H - 1) y1 = H - 1;
        if (x0 < 0) x0 = 0;
        if (x0 > W - 1) x0 = W - 1;
        if (x1 < 0) x1 = 0;
        if (x1 > W - 1) x1 = W - 1;

        // Fractional parts based on original src and unclamped floor (y0, x0)
        // Note: if y0 was clamped, the fractional part may be outside [0,1], but PyTorch still uses it;
        // however, if y0 and y1 become equal due to clamping, the weight sum still equals 1 because
        // (1-ly)+ly = 1. So we can compute ly = src_y - floorf(src_y) using original floor value.
        float ly = src_y - floorf(src_y);
        float lx = src_x - floorf(src_x);

        // Four corners (using clamped indices)
        float v00 = input[((n * C + c) * H + y0) * W + x0];
        float v01 = input[((n * C + c) * H + y0) * W + x1];
        float v10 = input[((n * C + c) * H + y1) * W + x0];
        float v11 = input[((n * C + c) * H + y1) * W + x1];

        // Bilinear interpolation
        float top = v00 * (1.0f - lx) + v01 * lx;
        float bottom = v10 * (1.0f - lx) + v11 * lx;
        float value = top * (1.0f - ly) + bottom * ly;

        output[i] = value;
    }
}

// Launcher called from C++ binding (extern "C")
// Parameters:
//   input: pointer to input tensor data (float32, contiguous)
//   output: pointer to output tensor data (pre-allocated, size N*C*outH*outW)
//   N, C, H, W: input dimensions
//   scale_factor: float scale factor (positive)
//   stream: CUDA stream
// Output dimensions computed as floor(H*scale) and floor(W*scale)
extern "C" void upsample_bilinear_2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    float scale_factor,
    cudaStream_t stream)
{
    if (scale_factor <= 0.0f || N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        return;
    }

    int outH = (int)(H * scale_factor);
    int outW = (int)(W * scale_factor);
    if (outH <= 0 || outW <= 0) {
        return;
    }

    int total = N * C * outH * outW;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    int max_grid = 65535;
    if (grid_size > max_grid) grid_size = max_grid;

    upsample_bilinear_2d_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, N, C, H, W, outH, outW, scale_factor);
}
