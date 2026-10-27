#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernels for constant_pad1d and nearest-neighbor interpolation
// All launchers use raw pointers and cudaStream_t, no PyTorch dependencies.

__global__ void constant_pad1d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t out_width,
    int64_t in_width,
    int64_t pad_left,
    int64_t pad_right,
    float value,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t in_stride_b,
    int64_t in_stride_c,
    int64_t in_stride_h,
    int64_t in_stride_w,
    int64_t out_stride_b,
    int64_t out_stride_c,
    int64_t out_stride_h,
    int64_t out_stride_w
) {
    int64_t total = batch * channels * height * out_width;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t w = i % out_width;
        int64_t tmp = i / out_width;
        int64_t h = tmp % height;
        tmp /= height;
        int64_t c = tmp % channels;
        int64_t b = tmp / channels;

        int64_t out_index = b * out_stride_b + c * out_stride_c + h * out_stride_h + w * out_stride_w;
        int64_t in_w = w - pad_left;
        if (in_w >= 0 && in_w < in_width) {
            int64_t in_index = b * in_stride_b + c * in_stride_c + h * in_stride_h + in_w * in_stride_w;
            output[out_index] = input[in_index];
        } else {
            output[out_index] = value;
        }
    }
}

__global__ void interpolate_nearest_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t output_height,
    int64_t output_width,
    int64_t input_height,
    int64_t input_width,
    int64_t batch,
    int64_t channels,
    int64_t in_stride_b,
    int64_t in_stride_c,
    int64_t in_stride_h,
    int64_t in_stride_w,
    int64_t out_stride_b,
    int64_t out_stride_c,
    int64_t out_stride_h,
    int64_t out_stride_w
) {
    int64_t total = batch * channels * output_height * output_width;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t ow = i % output_width;
        int64_t tmp = i / output_width;
        int64_t oh = tmp % output_height;
        tmp /= output_height;
        int64_t c = tmp % channels;
        int64_t b = tmp / channels;

        int64_t ih = (oh * input_height) / output_height;
        int64_t iw = (ow * input_width) / output_width;

        int64_t in_index = b * in_stride_b + c * in_stride_c + ih * in_stride_h + iw * in_stride_w;
        int64_t out_index = b * out_stride_b + c * out_stride_c + oh * out_stride_h + ow * out_stride_w;
        output[out_index] = input[in_index];
    }
}

extern "C" void constant_pad1d_launcher(
    float* output,
    const float* input,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t pad_left,
    int64_t pad_right,
    float value,
    int64_t in_stride_b,
    int64_t in_stride_c,
    int64_t in_stride_h,
    int64_t in_stride_w,
    int64_t out_stride_b,
    int64_t out_stride_c,
    int64_t out_stride_h,
    int64_t out_stride_w,
    cudaStream_t stream
) {
    int64_t out_width = width + pad_left + pad_right;
    int64_t total = batch * channels * height * out_width;
    if (total <= 0) return;

    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 1048576) blocks = 1048576; // grid-stride loop handles large sizes

    constant_pad1d_kernel<<<blocks, threads, 0, stream>>>(
        output, input, out_width, width, pad_left, pad_right, value,
        batch, channels, height,
        in_stride_b, in_stride_c, in_stride_h, in_stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error will be checked by binding layer; no throw here
    }
}

extern "C" void interpolate_nearest_launcher(
    float* output,
    const float* input,
    int64_t batch,
    int64_t channels,
    int64_t input_height,
    int64_t input_width,
    int64_t output_height,
    int64_t output_width,
    int64_t in_stride_b,
    int64_t in_stride_c,
    int64_t in_stride_h,
    int64_t in_stride_w,
    int64_t out_stride_b,
    int64_t out_stride_c,
    int64_t out_stride_h,
    int64_t out_stride_w,
    cudaStream_t stream
) {
    int64_t total = batch * channels * output_height * output_width;
    if (total <= 0) return;

    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 1048576) blocks = 1048576;

    interpolate_nearest_kernel<<<blocks, threads, 0, stream>>>(
        output, input, output_height, output_width, input_height, input_width,
        batch, channels,
        in_stride_b, in_stride_c, in_stride_h, in_stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error will be checked by binding layer
    }
}
