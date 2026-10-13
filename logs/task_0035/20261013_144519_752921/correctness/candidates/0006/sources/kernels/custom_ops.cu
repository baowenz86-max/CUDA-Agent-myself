#include <cuda_runtime.h>
#include <cstdint>

// Bilinear upsampling kernel, align_corners = false
// Input: NCHW float32 tensor, contiguous or with strides given by input strides
// Output: NCHW float32 tensor, contiguous
// scale_factor: float (can be non-integer)
// H_in, W_in: input spatial dims
// H_out, W_out: output spatial dims
__global__ void bilinear_upsample_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    float scale_factor,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = N * C * H_out * W_out;

    for (int idx = tid; idx < total; idx += stride) {
        // Decompose linear index to n,c,oy,ox
        int ox = idx % W_out;
        int tmp = idx / W_out;
        int oy = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        // Compute source coordinates using PyTorch's formula:
        // src = (dst + 0.5) * scale_factor - 0.5
        // Note: scale_factor is 1/scale = input_size / output_size, but here scale_factor is
        // the user provided scale_factor (e.g., 2.0). PyTorch uses scale = 1/scale_factor,
        // so src = (dst + 0.5) / scale_factor - 0.5.
        float src_x = (ox + 0.5f) / scale_factor - 0.5f;
        float src_y = (oy + 0.5f) / scale_factor - 0.5f;

        // PyTorch clamps the source coordinate to [-0.5, size-0.5] for interpolation,
        // but for index calculation it uses floor and ceil then clamps to [0, size-1].
        // We mimic that here:
        int x0 = (int)floorf(src_x);
        int y0 = (int)floorf(src_y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        // Clamp indices to valid range
        x0 = max(0, x0);
        y0 = max(0, y0);
        x1 = min(x1, W_in - 1);
        y1 = min(y1, H_in - 1);

        float x_frac = src_x - x0;
        float y_frac = src_y - y0;

        // Compute base pointer for this (n,c) plane using input strides
        const float* in_nc = input + n * input_stride_n + c * input_stride_c;

        // Read four neighbors using input strides for h,w
        float p00 = in_nc[y0 * input_stride_h + x0 * input_stride_w];
        float p01 = in_nc[y0 * input_stride_h + x1 * input_stride_w];
        float p10 = in_nc[y1 * input_stride_h + x0 * input_stride_w];
        float p11 = in_nc[y1 * input_stride_h + x1 * input_stride_w];

        // Bilinear interpolation
        float value = p00 * (1.0f - x_frac) * (1.0f - y_frac) +
                      p01 * x_frac * (1.0f - y_frac) +
                      p10 * (1.0f - x_frac) * y_frac +
                      p11 * x_frac * y_frac;

        output[idx] = value;
    }
}

// Logical AND threshold kernel: output bool = (x > 0) && (x < 1)
__global__ void logical_and_threshold_kernel(
    const float* __restrict__ input,
    bool* __restrict__ output,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        float val = input[i];
        output[i] = (val > 0.0f) && (val < 1.0f);
    }
}

// Launcher for bilinear upsampling
extern "C" void bilinear_upsample_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    float scale_factor,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    cudaStream_t stream)
{
    int total = N * C * H_out * W_out;
    if (total == 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    bilinear_upsample_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out, scale_factor,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w);

    cudaError_t err = cudaGetLastError();
    // No throw; binding checks after launch.
}

// Launcher for logical_and_threshold
extern "C" void logical_and_threshold_launcher(
    const float* input,
    bool* output,
    int total,
    cudaStream_t stream)
{
    if (total == 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    logical_and_threshold_kernel<<<blocks, threads, 0, stream>>>(input, output, total);

    cudaError_t err = cudaGetLastError();
    // No throw; binding checks after launch.
}
