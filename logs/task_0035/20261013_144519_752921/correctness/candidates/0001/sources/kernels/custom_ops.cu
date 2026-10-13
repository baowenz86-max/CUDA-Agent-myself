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
    long long input_stride_n,  // use long long to avoid overflow for large tensors
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

        // Compute source coordinates (align_corners=False)
        float src_x = (ox + 0.5f) / scale_factor - 0.5f;
        float src_y = (oy + 0.5f) / scale_factor - 0.5f;

        // Clamp to valid interpolation range: [-0.5, H_in-1+0.5] or [-0.5, W_in-1+0.5]
        src_x = fmaxf(-0.5f, fminf(src_x, W_in - 1 + 0.5f));
        src_y = fmaxf(-0.5f, fminf(src_y, H_in - 1 + 0.5f));

        // Determine integer and fractional parts
        int x1 = (int)floorf(src_x);
        int y1 = (int)floorf(src_y);
        float x_frac = src_x - x1;
        float y_frac = src_y - y1;

        // Clamp indices to [0, W_in-1] and [0, H_in-1] for actual reads
        int x0 = max(0, x1);
        int x2 = min(x1 + 1, W_in - 1);
        int y0 = max(0, y1);
        int y2 = min(y1 + 1, H_in - 1);

        // Compute base pointer for this (n,c) plane using input strides
        const float* in_nc = input + n * input_stride_n + c * input_stride_c;

        // Read four neighbors using input strides for h,w
        float p00 = in_nc[y0 * input_stride_h + x0 * input_stride_w];
        float p01 = in_nc[y0 * input_stride_h + x2 * input_stride_w];
        float p10 = in_nc[y2 * input_stride_h + x0 * input_stride_w];
        float p11 = in_nc[y2 * input_stride_h + x2 * input_stride_w];

        // Bilinear interpolation
        float value = p00 * (1 - x_frac) * (1 - y_frac) +
                      p01 * x_frac * (1 - y_frac) +
                      p10 * (1 - x_frac) * y_frac +
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
// Parameters: input pointer, output pointer, N, C, H_in, W_in, H_out, W_out, scale_factor, strides, stream
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
    // Cap blocks to reasonable maximum for grid-stride loop
    if (blocks > 65535) blocks = 65535;

    bilinear_upsample_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out, scale_factor,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a binding we would throw, here we just ignore or can set a flag.
        // But launcher cannot throw; it's up to the binding to check errors,
        // but we can perform a quick check to aid debugging.
        // We'll leave this as a no-op; the binding should call cudaGetLastError after launch.
    }
}

// Launcher for logical_and_threshold
// Parameters: input pointer, output bool pointer, total elements, stream
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
    // Same: no throw, but binding can check after launch
}
