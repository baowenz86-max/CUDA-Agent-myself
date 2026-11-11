#include <cuda_runtime.h>

// CUDA kernel for affine grid generation + scale
// theta: [N, 2, 3] float32, possibly non-contiguous
// output: [N, H, W, 2] float32, contiguous
// align_corners: if true, coordinates from -1 to 1 inclusive at corners;
//                else PyTorch convention: from -1 + 1/H to 1 - 1/H
// scale: scalar multiplier applied to both coordinates
template <bool ALIGN_CORNERS>
__global__ void affine_grid_scaled_kernel(
    const float* __restrict__ theta,
    float* __restrict__ output,
    int64_t N, int64_t H, int64_t W,
    int64_t theta_stride_n, int64_t theta_stride_r, int64_t theta_stride_c,
    float scale)
{
    int64_t total = N * H * W;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t w = i % W;
        int64_t tmp = i / W;
        int64_t h = tmp % H;
        int64_t n = tmp / H;

        // Compute base x, y from theta[n]
        const float* theta_n = theta + n * theta_stride_n;
        float x = theta_n[0 * theta_stride_r + 0 * theta_stride_c] * w +
                  theta_n[0 * theta_stride_r + 1 * theta_stride_c] * h +
                  theta_n[0 * theta_stride_r + 2 * theta_stride_c];
        float y = theta_n[1 * theta_stride_r + 0 * theta_stride_c] * w +
                  theta_n[1 * theta_stride_r + 1 * theta_stride_c] * h +
                  theta_n[1 * theta_stride_r + 2 * theta_stride_c];

        // Apply align_corners convention to get base_x, base_y
        float base_x, base_y;
        if (ALIGN_CORNERS) {
            base_x = (W == 1) ? 0.0f : (2.0f * w / (W - 1) - 1.0f);
            base_y = (H == 1) ? 0.0f : (2.0f * h / (H - 1) - 1.0f);
        } else {
            base_x = (W == 1) ? 0.0f : (2.0f * (w + 0.5f) / W - 1.0f);
            base_y = (H == 1) ? 0.0f : (2.0f * (h + 0.5f) / H - 1.0f);
        }

        // grid coordinate = base + theta linear transformation, then scale
        float out_x = (base_x * theta_n[0 * theta_stride_r + 0 * theta_stride_c] +
                       base_y * theta_n[0 * theta_stride_r + 1 * theta_stride_c] +
                       theta_n[0 * theta_stride_r + 2 * theta_stride_c]) * scale;
        float out_y = (base_x * theta_n[1 * theta_stride_r + 0 * theta_stride_c] +
                       base_y * theta_n[1 * theta_stride_r + 1 * theta_stride_c] +
                       theta_n[1 * theta_stride_r + 2 * theta_stride_c]) * scale;

        output[2 * i] = out_x;
        output[2 * i + 1] = out_y;
    }
}

// Launcher function (C interface)
extern "C" void affine_grid_scaled_launcher(
    const float* theta,
    float* output,
    int64_t N, int64_t H, int64_t W,
    int64_t theta_stride_n, int64_t theta_stride_r, int64_t theta_stride_c,
    bool align_corners,
    float scale,
    cudaStream_t stream)
{
    if (N <= 0 || H <= 0 || W <= 0) return;

    int64_t total = N * H * W;
    int block = 256;
    int64_t grid = (total + block - 1) / block;
    if (grid > 2147483647LL) grid = 2147483647LL;

    if (align_corners) {
        affine_grid_scaled_kernel<true><<<(unsigned int)grid, block, 0, stream>>>(
            theta, output, N, H, W, theta_stride_n, theta_stride_r, theta_stride_c, scale);
    } else {
        affine_grid_scaled_kernel<false><<<(unsigned int)grid, block, 0, stream>>>(
            theta, output, N, H, W, theta_stride_n, theta_stride_r, theta_stride_c, scale);
    }
}
