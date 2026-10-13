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
        int ox = idx % W_out;
        int tmp = idx / W_out;
        int oy = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        // Compute source coordinates (align_corners=False)
        float src_x = (ox + 0.5f) / scale_factor - 0.5f;
        float src_y = (oy + 0.5f) / scale_factor - 0.5f;

        // Clamp to valid interpolation range: [-0.5, W_in-1+0.5] and [-0.5, H_in-1+0.5]
        src_x = fmaxf(-0.5f, fminf(src_x, W_in - 1 + 0.5f));
        src_y = fmaxf(-0.5f, fminf(src_y, H_in - 1 + 0.5f));

        // Determine neighbor indices exactly as PyTorch does:
        // x0 = max(0, floor(src_x))
        // x1 = min(x0 + 1, W_in - 1)
        // lambda_x = src_x - x0, but ensure lambda_x in [0,1]
        int x0 = max(0, static_cast<int>(floorf(src_x)));
        int x1 = min(x0 + 1, W_in - 1);
        float lambda_x = src_x - x0;
        lambda_x = fmaxf(0.0f, fminf(lambda_x, 1.0f)); // for safety

        int y0 = max(0, static_cast<int>(floorf(src_y)));
        int y1 = min(y0 + 1, H_in - 1);
        float lambda_y = src_y - y0;
        lambda_y = fmaxf(0.0f, fminf(lambda_y, 1.0f)); // for safety

        // Compute base pointer for this (n,c) plane
        const float* in_nc = input + n * input_stride_n + c * input_stride_c;

        // Read four neighbors
        float p00 = in_nc[y0 * input_stride_h + x0 * input_stride_w];
        float p01 = in_nc[y0 * input_stride_h + x1 * input_stride_w];
        float p10 = in_nc[y1 * input_stride_h + x0 * input_stride_w];
        float p11 = in_nc[y1 * input_stride_h + x1 * input_stride_w];

        // Bilinear interpolation
        float value = p00 * (1.0f - lambda_x) * (1.0f - lambda_y) +
                      p01 * lambda_x * (1.0f - lambda_y) +
                      p10 * (1.0f - lambda_x) * lambda_y +
                      p11 * lambda_x * lambda_y;

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
    // ignore in launcher, binding checks after
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
}
