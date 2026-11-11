#include <cuda_runtime.h>

// Kernel: sum over dimension 1 (channels) with keepdim=True
// Input: [B, C, H, W], Output: [B, 1, H, W]
// Supports non-contiguous input via strides (assumes int64_t strides, but we use int64_t for safety)
__global__ void sum_dim1_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t in_stride_b,
    int64_t in_stride_c,
    int64_t in_stride_h,
    int64_t in_stride_w,
    int64_t out_stride_b,
    int64_t out_stride_c,
    int64_t out_stride_h,
    int64_t out_stride_w
) {
    int64_t total_hw = height * width;
    int64_t total = batch_size * total_hw;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        int64_t b = i / total_hw;
        int64_t hw = i % total_hw;
        int64_t h = hw / width;
        int64_t w = hw % width;

        float sum = 0.0f;
        const float* in_base = input + b * in_stride_b + h * in_stride_h + w * in_stride_w;
        for (int64_t c = 0; c < channels; ++c) {
            sum += in_base[c * in_stride_c];
        }
        float* out_ptr = output + b * out_stride_b + h * out_stride_h + w * out_stride_w;
        *out_ptr = sum;
    }
}

// Kernel: element-wise leaky relu
__global__ void leaky_relu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel,
    float negative_slope
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = val >= 0.0f ? val : negative_slope * val;
    }
}

// Launcher for sum_dim with dim=1, keepdim=true (generalized but only used for dim=1)
extern "C" void sum_dim_launcher(
    float* output,
    const float* input,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
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
    int64_t total = batch_size * height * width;
    if (total == 0) return;
    int threads = 256;
    int64_t blocks64 = (total + threads - 1) / threads;
    // grid-stride loop handles any grid size; cap to reasonable max
    if (blocks64 > 65535) blocks64 = 65535;
    int blocks = (int)blocks64;
    sum_dim1_kernel<<<blocks, threads, 0, stream>>>(
        output, input,
        batch_size, channels, height, width,
        in_stride_b, in_stride_c, in_stride_h, in_stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w
    );
}

// Launcher for leaky_relu element-wise (input assumed contiguous, but can handle if caller passes appropriate pointer)
extern "C" void leaky_relu_launcher(
    float* output,
    const float* input,
    int64_t numel,
    float negative_slope,
    cudaStream_t stream
) {
    if (numel == 0) return;
    int threads = 256;
    int64_t blocks64 = (numel + threads - 1) / threads;
    if (blocks64 > 65535) blocks64 = 65535;
    int blocks = (int)blocks64;
    leaky_relu_kernel<<<blocks, threads, 0, stream>>>(
        output, input, numel, negative_slope
    );
}
