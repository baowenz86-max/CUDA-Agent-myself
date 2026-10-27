#include <cuda_runtime.h>
#include <cstdint>

// Bilinear upsampling kernel (align_corners=false)
// Computes output value for each output spatial location (n, c, oh, ow)
// Input layout: [N, C, H, W], strides: isN, isC, isH, isW (elements)
// Output layout: [N, C, OH, OW]
template <typename scalar_t>
__global__ void upsample_bilinear2d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t OH, int64_t OW,
    int64_t isN, int64_t isC, int64_t isH, int64_t isW,
    int64_t osN, int64_t osC, int64_t osH, int64_t osW,
    double scale_factor) {
    int64_t total = N * C * OH * OW;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        // decompose idx into output coordinates (n, c, oh, ow)
        int64_t ow = idx % OW;
        int64_t tmp = idx / OW;
        int64_t oh = tmp % OH;
        tmp /= OH;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // source coordinate for align_corners=false
        // src = (dst + 0.5) / scale - 0.5
        double h_src = (static_cast<double>(oh) + 0.5) / scale_factor - 0.5;
        double w_src = (static_cast<double>(ow) + 0.5) / scale_factor - 0.5;

        // clamp to [0, H-1]
        if (h_src < 0) h_src = 0;
        if (w_src < 0) w_src = 0;
        if (h_src > static_cast<double>(H - 1)) h_src = static_cast<double>(H - 1);
        if (w_src > static_cast<double>(W - 1)) w_src = static_cast<double>(W - 1);

        int64_t h0 = static_cast<int64_t>(h_src);
        int64_t w0 = static_cast<int64_t>(w_src);
        int64_t h1 = h0 + 1;
        int64_t w1 = w0 + 1;
        if (h1 >= H) h1 = H - 1;
        if (w1 >= W) w1 = W - 1;

        double h1_lambda = h_src - static_cast<double>(h0);
        double w1_lambda = w_src - static_cast<double>(w0);
        double h0_lambda = 1.0 - h1_lambda;
        double w0_lambda = 1.0 - w1_lambda;

        // gather input values with strides
        const scalar_t* base = input + n * isN + c * isC;
        scalar_t v00 = base[h0 * isH + w0 * isW];
        scalar_t v01 = base[h0 * isH + w1 * isW];
        scalar_t v10 = base[h1 * isH + w0 * isW];
        scalar_t v11 = base[h1 * isH + w1 * isW];

        double result = static_cast<double>(v00) * h0_lambda * w0_lambda +
                        static_cast<double>(v01) * h0_lambda * w1_lambda +
                        static_cast<double>(v10) * h1_lambda * w0_lambda +
                        static_cast<double>(v11) * h1_lambda * w1_lambda;

        output[n * osN + c * osC + oh * osH + ow * osW] = static_cast<scalar_t>(result);
    }
}

extern "C" void upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t OH, int64_t OW,
    int64_t isN, int64_t isC, int64_t isH, int64_t isW,
    int64_t osN, int64_t osC, int64_t osH, int64_t osW,
    double scale_factor,
    cudaStream_t stream) {
    int64_t total = N * C * OH * OW;
    if (total <= 0) return;

    const int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535; // clamp for grid-stride loop

    upsample_bilinear2d_kernel<float><<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input, output, N, C, H, W, OH, OW,
        isN, isC, isH, isW,
        osN, osC, osH, osW,
        scale_factor);

    // Capture launch error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // We cannot throw here; binding layer can check later. But we can at least assert in debug builds.
        // For now, simply ignore as binding layer will synchronize and check.
    }
}
