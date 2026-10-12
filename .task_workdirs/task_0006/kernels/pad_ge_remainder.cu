#include <cuda_runtime.h>
#include <cstdint>

extern "C" void pad_ge_remainder_forward_launcher(
    int64_t* output,
    const float* input,
    int batch_size,
    int channels,
    int height,
    int width,
    int pad_left,
    int pad_right,
    int pad_top,
    int pad_bottom,
    cudaStream_t stream);

__global__ void pad_ge_remainder_kernel(
    int64_t* __restrict__ output,
    const float* __restrict__ input,
    int batch_size,
    int channels,
    int height,
    int width,
    int out_height,
    int out_width,
    int pad_left,
    int pad_top) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = static_cast<int64_t>(batch_size) * channels * out_height * out_width;
    if (idx >= total_elements) {
        return;
    }

    // Decompose linear index
    int ow = idx % out_width;
    int oh = (idx / out_width) % out_height;
    int oc = (idx / (out_width * out_height)) % channels;
    int ob = idx / (out_width * out_height * channels);

    // Reflection padding coordinate
    int ih = oh - pad_top;
    if (ih < 0) ih = -ih;
    if (ih >= height) ih = 2 * height - 2 - ih;

    int iw = ow - pad_left;
    if (iw < 0) iw = -iw;
    if (iw >= width) iw = 2 * width - 2 - iw;

    float val = input[((ob * channels + oc) * height + ih) * width + iw];

    // Compute (val >= 0)
    bool ge = (val >= 0.0f);
    // Compute remainder by 2 (result is 0.0 or 1.0)
    float rem = (ge ? 1.0f : 0.0f);
    rem = fmodf(rem, 2.0f);

    output[idx] = (int64_t)rem;
}

extern "C" void pad_ge_remainder_forward_launcher(
    int64_t* output,
    const float* input,
    int batch_size,
    int channels,
    int height,
    int width,
    int pad_left,
    int pad_right,
    int pad_top,
    int pad_bottom,
    cudaStream_t stream) {
    int64_t out_height = static_cast<int64_t>(height) + pad_top + pad_bottom;
    int64_t out_width = static_cast<int64_t>(width) + pad_left + pad_right;
    int64_t total_elements = static_cast<int64_t>(batch_size) * channels * out_height * out_width;

    if (total_elements == 0) {
        return;
    }

    const int block_size = 256;
    int64_t grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 65535) {
        grid_size = 65535;
    }

    pad_ge_remainder_kernel<<<static_cast<int>(grid_size), block_size, 0, stream>>>(
        output,
        input,
        batch_size,
        channels,
        height,
        width,
        static_cast<int>(out_height),
        static_cast<int>(out_width),
        pad_left,
        pad_top);
}
