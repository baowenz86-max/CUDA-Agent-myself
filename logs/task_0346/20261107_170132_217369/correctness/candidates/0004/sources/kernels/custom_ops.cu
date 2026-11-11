#include <cuda_runtime.h>

// Bilinear upsampling kernel (align_corners=False)
// Input: float32 tensor of shape [N, C, H, W], contiguous
// Output: float32 tensor of shape [N, C, outH, outW]
// scale_factor: float, output size = floor(input_size * scale_factor)
// This kernel uses a 2D grid over output pixels to avoid complex index decomposition.
__global__ void upsample_bilinear_2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    float scale_factor)
{
    // Map blockIdx and threadIdx to output pixel (ow, oh) and batch/channel
    // Grid dimensions: x = outW, y = outH, z = N*C (or we can use blockIdx.z)
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int nc = blockIdx.z; // nc = n*C + c, but we need to map to n and c

    if (ow >= outW || oh >= outH || nc >= N * C) return;

    int n = nc / C;
    int c = nc - n * C;

    // Source coordinate mapping: src = (dst + 0.5) / scale - 0.5
    double src_y = (oh + 0.5) / (double)scale_factor - 0.5;
    double src_x = (ow + 0.5) / (double)scale_factor - 0.5;

    // Clamp to input bounds to avoid extrapolation
    if (src_y < 0.0) src_y = 0.0;
    if (src_x < 0.0) src_x = 0.0;
    if (src_y > H - 1.0) src_y = H - 1.0;
    if (src_x > W - 1.0) src_x = W - 1.0;

    int y0 = (int)floor((float)src_y);
    int x0 = (int)floor((float)src_x);
    int y1 = y0 + 1;
    int x1 = x0 + 1;
    if (y1 >= H) y1 = H - 1;
    if (x1 >= W) x1 = W - 1;

    float ly = (float)(src_y - y0);
    float lx = (float)(src_x - x0);

    // Four corners
    const float* base = input + ((n * C + c) * H) * W; // pointer to the input slice
    float v00 = base[y0 * W + x0];
    float v01 = base[y0 * W + x1];
    float v10 = base[y1 * W + x0];
    float v11 = base[y1 * W + x1];

    // Bilinear interpolation
    float top = v00 * (1.0f - lx) + v01 * lx;
    float bottom = v10 * (1.0f - lx) + v11 * lx;
    float value = top * (1.0f - ly) + bottom * ly;

    output[((n * C + c) * outH + oh) * outW + ow] = value;
}

// Launcher called from C++ binding (extern "C")
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

    // Use 2D block and 3D grid for clarity
    dim3 block(16, 16); // 256 threads per block
    dim3 grid((outW + block.x - 1) / block.x,
              (outH + block.y - 1) / block.y,
              N * C); // one block z per slice, up to 65535 slices in z, but N*C=48, safe

    upsample_bilinear_2d_kernel<<<grid, block, 0, stream>>>(
        input, output, N, C, H, W, outH, outW, scale_factor);
}
