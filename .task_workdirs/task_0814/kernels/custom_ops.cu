#include <cuda_runtime.h>

// Bilinear upsampling with scale factor 2, align_corners=False.
// Input: contiguous float32 tensor [N, C, H, W]
// Output: contiguous float32 tensor [N, C, 2H, 2W]
// Formula: src = (dst + 0.5) / 2 - 0.5, clamp to [0, size-1], bilinear interpolation.
__global__ void upsample_bilinear_scale2_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t outH, int64_t outW,
    int64_t total_elements) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t linear_idx = idx; linear_idx < total_elements; linear_idx += stride) {
        // Decompose linear index to (n, c, oh, ow)
        int64_t tmp = linear_idx;
        int64_t ow = tmp % outW;
        tmp /= outW;
        int64_t oh = tmp % outH;
        tmp /= outH;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // Source coordinates for align_corners=False
        float src_x = (ow + 0.5f) * 0.5f - 0.5f;
        float src_y = (oh + 0.5f) * 0.5f - 0.5f;

        // Clamp to [0, W-1] and [0, H-1]
        src_x = fmaxf(0.0f, fminf(src_x, (float)(W - 1)));
        src_y = fmaxf(0.0f, fminf(src_y, (float)(H - 1)));

        int x0 = (int)floorf(src_x);
        int y0 = (int)floorf(src_y);
        int x1 = min(x0 + 1, (int)(W - 1));
        int y1 = min(y0 + 1, (int)(H - 1));

        float dx = src_x - x0;
        float dy = src_y - y0;

        // Base pointer to first row (y0) and x0
        const float* row0 = input + ((n * C + c) * H + y0) * W + x0;
        const float* row1 = input + ((n * C + c) * H + y1) * W + x0;

        // Read the four values; when x0 == x1, the second read is same as first, but within bounds.
        float v00 = row0[0];
        float v01 = (x1 > x0) ? row0[1] : row0[0];
        float v10 = row1[0];
        float v11 = (x1 > x0) ? row1[1] : row1[0];

        // Bilinear interpolation
        float val = (v00 * (1.0f - dx) + v01 * dx) * (1.0f - dy)
                  + (v10 * (1.0f - dx) + v11 * dx) * dy;

        output[linear_idx] = val;
    }
}

// Constant padding for 4D tensor.
// Input: [N, C, H, W], output: [N, C, H + top + bottom, W + left + right]
__global__ void constant_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t left, int64_t right, int64_t top, int64_t bottom,
    float value,
    int64_t outH, int64_t outW,
    int64_t total_elements) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t linear_idx = idx; linear_idx < total_elements; linear_idx += stride) {
        int64_t tmp = linear_idx;
        int64_t ow = tmp % outW;
        tmp /= outW;
        int64_t oh = tmp % outH;
        tmp /= outH;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        if (oh < top || oh >= top + H || ow < left || ow >= left + W) {
            output[linear_idx] = value;
        } else {
            int64_t ih = oh - top;
            int64_t iw = ow - left;
            output[linear_idx] = input[((n * C + c) * H + ih) * W + iw];
        }
    }
}

extern "C" void upsample_bilinear_scale2_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    cudaStream_t stream) {

    int64_t outH = 2 * H;
    int64_t outW = 2 * W;
    int64_t total = N * C * outH * outW;

    const int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    // Cap grid at something reasonable; use grid-stride loop
    if (blocks > 65535) blocks = 65535;

    upsample_bilinear_scale2_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, outH, outW, total);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In launcher we cannot throw; but we can print if needed (avoid printing in production)
        // We leave error handling to binding layer, but check here for debugging.
    }
}

extern "C" void constant_pad2d_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t left, int64_t right, int64_t top, int64_t bottom,
    double value,
    cudaStream_t stream) {

    int64_t outH = H + top + bottom;
    int64_t outW = W + left + right;
    int64_t total = N * C * outH * outW;

    const int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    constant_pad2d_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, left, right, top, bottom,
        (float)value, outH, outW, total);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error will be caught by binding layer.
    }
}
