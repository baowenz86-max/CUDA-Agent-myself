#include <cuda_runtime.h>
#include <cstdint>

// Helper to compute index given strides
__device__ __forceinline__ int64_t flat_index(int64_t b, int64_t c, int64_t h, int64_t w,
                                                int64_t stride_b, int64_t stride_c, int64_t stride_h, int64_t stride_w) {
    return b * stride_b + c * stride_c + h * stride_h + w * stride_w;
}

// Bilinear interpolation kernel, align_corners=False.
// Each thread computes one output pixel.
__global__ void interpolate_bilinear_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t in_batch, int64_t in_channels, int64_t in_height, int64_t in_width,
    int64_t out_height, int64_t out_width,
    int64_t in_stride_b, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int64_t out_stride_b, int64_t out_stride_c, int64_t out_stride_h, int64_t out_stride_w,
    float scale) {
    int64_t total = in_batch * in_channels * out_height * out_width;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t ow = i % out_width;
        int64_t oh = (i / out_width) % out_height;
        int64_t oc = (i / (out_width * out_height)) % in_channels;
        int64_t ob = i / (out_width * out_height * in_channels);

        // align_corners=False mapping: src_x = (dst_x + 0.5) / scale - 0.5
        float src_h = (oh + 0.5f) / scale - 0.5f;
        float src_w = (ow + 0.5f) / scale - 0.5f;

        // Clamp to [0, in-1] for border handling (PyTorch default for bilinear is no clamp, but out of range values use border? Actually PyTorch uses border mode: indices outside are clamped)
        // PyTorch's bilinear interpolation with align_corners=False clamps source coordinates to [0, size-1] if they fall outside.
        src_h = fmaxf(0.0f, fminf(src_h, (float)(in_height - 1)));
        src_w = fmaxf(0.0f, fminf(src_w, (float)(in_width - 1)));

        int h0 = (int)floorf(src_h);
        int w0 = (int)floorf(src_w);
        int h1 = min(h0 + 1, (int)(in_height - 1));
        int w1 = min(w0 + 1, (int)(in_width - 1));
        float h_lambda = src_h - h0;
        float w_lambda = src_w - w0;

        // Read four neighbors using input strides
        int64_t base = flat_index(ob, oc, 0, 0, in_stride_b, in_stride_c, in_stride_h, in_stride_w);
        const float* row0 = input + base + h0 * in_stride_h;
        const float* row1 = input + base + h1 * in_stride_h;

        float v00 = row0[w0 * in_stride_w];
        float v01 = row0[w1 * in_stride_w];
        float v10 = row1[w0 * in_stride_w];
        float v11 = row1[w1 * in_stride_w];

        float top = v00 * (1.0f - w_lambda) + v01 * w_lambda;
        float bottom = v10 * (1.0f - w_lambda) + v11 * w_lambda;
        float result = top * (1.0f - h_lambda) + bottom * h_lambda;

        int64_t out_idx = flat_index(ob, oc, oh, ow, out_stride_b, out_stride_c, out_stride_h, out_stride_w);
        output[out_idx] = result;
    }
}

// Sum over last two dimensions (dim=2 and dim=3).
// Each thread computes one output element (batch, channel), looping over H and W.
__global__ void sum_reduction_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t batch, int64_t channels, int64_t height, int64_t width,
    int64_t in_stride_b, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int64_t out_stride_b, int64_t out_stride_c) {
    int64_t total_out = batch * channels;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_out; i += stride) {
        int64_t c = i % channels;
        int64_t b = i / channels;

        float sum = 0.0f;
        for (int64_t h = 0; h < height; ++h) {
            for (int64_t w = 0; w < width; ++w) {
                int64_t in_idx = flat_index(b, c, h, w, in_stride_b, in_stride_c, in_stride_h, in_stride_w);
                sum += input[in_idx];
            }
        }
        int64_t out_idx = b * out_stride_b + c * out_stride_c;
        output[out_idx] = sum;
    }
}

extern "C" {

void interpolate_bilinear_launcher(
    const float* input,
    float* output,
    int64_t in_batch, int64_t in_channels, int64_t in_height, int64_t in_width,
    int64_t out_height, int64_t out_width,
    int64_t in_stride_b, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int64_t out_stride_b, int64_t out_stride_c, int64_t out_stride_h, int64_t out_stride_w,
    double scale_factor,
    cudaStream_t stream) {

    float scale = (float)scale_factor;
    int64_t total = in_batch * in_channels * out_height * out_width;
    int block = 256;
    int grid = (int)min((total + block - 1) / block, (int64_t)2147483647); // clamp grid
    if (grid < 1) grid = 1;
    interpolate_bilinear_kernel<<<grid, block, 0, stream>>>(
        input, output,
        in_batch, in_channels, in_height, in_width,
        out_height, out_width,
        in_stride_b, in_stride_c, in_stride_h, in_stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w,
        scale);
}

void sum_reduction_launcher(
    const float* input,
    float* output,
    int64_t batch, int64_t channels, int64_t height, int64_t width,
    int64_t in_stride_b, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int64_t out_stride_b, int64_t out_stride_c,
    cudaStream_t stream) {

    int64_t total_out = batch * channels;
    int block = 256;
    int grid = (int)min((total_out + block - 1) / block, (int64_t)2147483647);
    if (grid < 1) grid = 1;
    sum_reduction_kernel<<<grid, block, 0, stream>>>(
        input, output,
        batch, channels, height, width,
        in_stride_b, in_stride_c, in_stride_h, in_stride_w,
        out_stride_b, out_stride_c);
}

} // extern "C"
