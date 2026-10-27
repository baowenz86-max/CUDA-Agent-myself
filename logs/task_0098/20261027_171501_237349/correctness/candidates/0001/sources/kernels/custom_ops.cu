#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernels for reflection_pad2d, elementwise maximum, and elementwise minimum.
// All operate on float32 tensors. Launchers assume contiguous inputs and outputs.
// Binding layer should allocate output tensor contiguous and call these functions.

template <typename T>
__device__ __forceinline__ T reflect_index(T idx, T size) {
    // Reflection padding: index maps to [0, size-1]
    if (idx < 0) idx = -idx;
    if (idx >= size) idx = 2 * size - 2 - idx;
    // The above may still be out of range if pad is larger than size-1,
    // but for usual cases pad <= size-1. For safety, clamp.
    if (idx < 0) idx = 0;
    if (idx >= size) idx = size - 1;
    return idx;
}

__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t batch,
    int64_t channels,
    int64_t in_h,
    int64_t in_w,
    int64_t out_h,
    int64_t out_w,
    int pad_size
) {
    int64_t total = batch * channels * out_h * out_w;
    int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = index; i < total; i += stride) {
        int64_t ow = i % out_w;
        int64_t tmp = i / out_w;
        int64_t oh = tmp % out_h;
        tmp /= out_h;
        int64_t c = tmp % channels;
        int64_t b = tmp / channels;

        int64_t ih = oh - pad_size;
        int64_t iw = ow - pad_size;
        ih = reflect_index(ih, in_h);
        iw = reflect_index(iw, in_w);

        int64_t input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
        output[i] = input[input_idx];
    }
}

__global__ void maximum_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t numel
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = i; idx < numel; idx += stride) {
        out[idx] = fmaxf(a[idx], b[idx]);
    }
}

__global__ void minimum_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t numel
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t idx = i; idx < numel; idx += stride) {
        out[idx] = fminf(a[idx], b[idx]);
    }
}

// Launcher functions with C linkage, to be called from binding.cpp
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int64_t batch,
    int64_t channels,
    int64_t in_h,
    int64_t in_w,
    int pad_size,
    cudaStream_t stream
) {
    int64_t out_h = in_h + 2 * pad_size;
    int64_t out_w = in_w + 2 * pad_size;
    int64_t total = batch * channels * out_h * out_w;

    const int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;

    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch, channels, in_h, in_w, out_h, out_w, pad_size);
}

extern "C" void maximum_launcher(
    const float* a,
    const float* b,
    float* out,
    int64_t numel,
    cudaStream_t stream
) {
    const int threads = 256;
    int64_t blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;

    maximum_kernel<<<blocks, threads, 0, stream>>>(a, b, out, numel);
}

extern "C" void minimum_launcher(
    const float* a,
    const float* b,
    float* out,
    int64_t numel,
    cudaStream_t stream
) {
    const int threads = 256;
    int64_t blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    if (blocks < 1) blocks = 1;

    minimum_kernel<<<blocks, threads, 0, stream>>>(a, b, out, numel);
}
