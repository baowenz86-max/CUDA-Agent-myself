#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernels and launchers for bilinear 2x upsampling and adaptive average pooling.
// No PyTorch dependencies are used. Binding layer will call these extern "C" launchers.

// -----------------------------------------------------------------------------
// Bilinear upsampling with scale_factor (integer) for 4D tensor [N, C, H, W]
// align_corners = false, half-pixel centers.
// input may be non-contiguous; strides are provided separately.
// -----------------------------------------------------------------------------
__global__ void bilinear_upsample_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int out_H, int out_W,
    int scale_factor,
    int64_t in_n_stride, int64_t in_c_stride, int64_t in_h_stride, int64_t in_w_stride,
    int64_t out_n_stride, int64_t out_c_stride, int64_t out_h_stride, int64_t out_w_stride)
{
    int64_t total_elements = (int64_t)N * C * out_H * out_W;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    float scale = static_cast<float>(scale_factor);

    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        // decompose linear index: n, c, oh, ow (row-major over output)
        int ow = static_cast<int>(linear % out_W);
        int64_t tmp = linear / out_W;
        int oh = static_cast<int>(tmp % out_H);
        tmp /= out_H;
        int c = static_cast<int>(tmp % C);
        int n = static_cast<int>(tmp / C);

        // half-pixel mapping: in_coordinate = (out_coordinate + 0.5) / scale - 0.5
        float x = (static_cast<float>(ow) + 0.5f) / scale - 0.5f;
        float y = (static_cast<float>(oh) + 0.5f) / scale - 0.5f;

        // clamp to [0, H-1] and [0, W-1] for sampling
        if (x < 0.0f) x = 0.0f;
        if (y < 0.0f) y = 0.0f;
        if (x > static_cast<float>(W - 1)) x = static_cast<float>(W - 1);
        if (y > static_cast<float>(H - 1)) y = static_cast<float>(H - 1);

        int x0 = static_cast<int>(floorf(x));
        int y0 = static_cast<int>(floorf(y));
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        // clamp indices
        if (x1 >= W) x1 = W - 1;
        if (y1 >= H) y1 = H - 1;

        float fx = x - static_cast<float>(x0);
        float fy = y - static_cast<float>(y0);

        const float* in_base = input + n * in_n_stride + c * in_c_stride;
        float v00 = in_base[y0 * in_h_stride + x0 * in_w_stride];
        float v01 = in_base[y0 * in_h_stride + x1 * in_w_stride];
        float v10 = in_base[y1 * in_h_stride + x0 * in_w_stride];
        float v11 = in_base[y1 * in_h_stride + x1 * in_w_stride];

        float top = v00 * (1.0f - fx) + v01 * fx;
        float bottom = v10 * (1.0f - fx) + v11 * fx;
        float value = top * (1.0f - fy) + bottom * fy;

        output[n * out_n_stride + c * out_c_stride + oh * out_h_stride + ow * out_w_stride] = value;
    }
}

extern "C" void interpolate_bilinear_scale_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int scale_factor,
    int64_t in_n_stride, int64_t in_c_stride, int64_t in_h_stride, int64_t in_w_stride,
    int64_t out_n_stride, int64_t out_c_stride, int64_t out_h_stride, int64_t out_w_stride,
    cudaStream_t stream)
{
    int out_H = H * scale_factor;
    int out_W = W * scale_factor;
    int64_t total_elements = (int64_t)N * C * out_H * out_W;

    const int block_size = 256;
    int64_t blocks64 = (total_elements + block_size - 1) / block_size;
    // Limit grid dimension to a reasonable maximum, but use grid-stride loop to handle rest
    int blocks = static_cast<int>(min(blocks64, (int64_t)65535));
    if (blocks == 0) blocks = 1;

    bilinear_upsample_kernel<<<blocks, block_size, 0, stream>>>(
        input, output, N, C, H, W, out_H, out_W, scale_factor,
        in_n_stride, in_c_stride, in_h_stride, in_w_stride,
        out_n_stride, out_c_stride, out_h_stride, out_w_stride);
}

// -----------------------------------------------------------------------------
// Adaptive average pooling for 4D tensor [N, C, H, W] to [N, C, out_h, out_w]
// Mirrors PyTorch's adaptive average pooling: for each output position (oh, ow),
// compute start and end indices in input:
//   start_index = floor(oh * H / out_h)
//   end_index   = ceil((oh+1) * H / out_h)
// similarly for width, then average all values in that window.
// -----------------------------------------------------------------------------
__global__ void adaptive_avg_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int out_h, int out_w,
    int64_t in_n_stride, int64_t in_c_stride, int64_t in_h_stride, int64_t in_w_stride,
    int64_t out_n_stride, int64_t out_c_stride, int64_t out_h_stride, int64_t out_w_stride)
{
    int64_t total_elements = (int64_t)N * C * out_h * out_w;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        // decompose output index: n, c, oh, ow
        int ow = static_cast<int>(linear % out_w);
        int64_t tmp = linear / out_w;
        int oh = static_cast<int>(tmp % out_h);
        tmp /= out_h;
        int c = static_cast<int>(tmp % C);
        int n = static_cast<int>(tmp / C);

        int h_start = (oh * H) / out_h;
        int h_end = ((oh + 1) * H + out_h - 1) / out_h;  // ceil division
        int w_start = (ow * W) / out_w;
        int w_end = ((ow + 1) * W + out_w - 1) / out_w;

        float sum = 0.0f;
        int count = 0;

        const float* in_base = input + n * in_n_stride + c * in_c_stride;
        for (int ih = h_start; ih < h_end; ++ih) {
            for (int iw = w_start; iw < w_end; ++iw) {
                sum += in_base[ih * in_h_stride + iw * in_w_stride];
                ++count;
            }
        }

        float avg = (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;
        output[n * out_n_stride + c * out_c_stride + oh * out_h_stride + ow * out_w_stride] = avg;
    }
}

extern "C" void adaptive_avg_pool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int out_h, int out_w,
    int64_t in_n_stride, int64_t in_c_stride, int64_t in_h_stride, int64_t in_w_stride,
    int64_t out_n_stride, int64_t out_c_stride, int64_t out_h_stride, int64_t out_w_stride,
    cudaStream_t stream)
{
    int64_t total_elements = (int64_t)N * C * out_h * out_w;
    if (total_elements == 0) return;

    const int block_size = 256;
    int64_t blocks64 = (total_elements + block_size - 1) / block_size;
    int blocks = static_cast<int>(min(blocks64, (int64_t)65535));
    if (blocks == 0) blocks = 1;

    adaptive_avg_pool2d_kernel<<<blocks, block_size, 0, stream>>>(
        input, output, N, C, H, W, out_h, out_w,
        in_n_stride, in_c_stride, in_h_stride, in_w_stride,
        out_n_stride, out_c_stride, out_h_stride, out_w_stride);
}
