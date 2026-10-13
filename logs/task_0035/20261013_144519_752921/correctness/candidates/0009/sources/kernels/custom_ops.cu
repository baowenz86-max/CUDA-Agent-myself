#include <cuda_runtime.h>
#include <cstdint>

// Bilinear upsampling kernel, align_corners = false
// Input: NCHW float32 tensor, contiguous or with strides given by input strides
// Output: NCHW float32 tensor, contiguous
// scale_factor: double (can be non-integer)
// H_in, W_in: input spatial dims
// H_out, W_out: output spatial dims
__global__ void bilinear_upsample_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    double scale_factor,
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

        // Compute source coordinates (align_corners=False) using double
        double src_x = (static_cast<double>(ox) + 0.5) / scale_factor - 0.5;
        double src_y = (static_cast<double>(oy) + 0.5) / scale_factor - 0.5;

        // Clamp to valid interpolation range: [-0.5, H_in-1+0.5] or [-0.5, W_in-1+0.5]
        src_x = fmax(-0.5, fmin(src_x, static_cast<double>(W_in - 1) + 0.5));
        src_y = fmax(-0.5, fmin(src_y, static_cast<double>(H_in - 1) + 0.5));

        // Determine integer and fractional parts using double floor/ceil
        double x1_floor = floor(src_x);
        double y1_floor = floor(src_y);
        double x_frac = src_x - x1_floor;
        double y_frac = src_y - y1_floor;

        // Clamp indices to [0, W_in-1] and [0, H_in-1] for actual reads
        int x0 = max(0, static_cast<int>(x1_floor));
        int y0 = max(0, static_cast<int>(y1_floor));
        int x1 = min(static_cast<int>(ceil(src_x)), W_in - 1);
        int y1 = min(static_cast<int>(ceil(src_y)), H_in - 1);

        // Compute base pointer for this (n,c) plane using input strides
        const float* in_nc = input + n * input_stride_n + c * input_stride_c;

        // Read four neighbors using input strides for h,w
        float p00 = in_nc[y0 * input_stride_h + x0 * input_stride_w];
        float p01 = in_nc[y0 * input_stride_h + x1 * input_stride_w];
        float p10 = in_nc[y1 * input_stride_h + x0 * input_stride_w];
        float p11 = in_nc[y1 * input_stride_h + x1 * input_stride_w];

        // Bilinear interpolation using double weights
        double value = static_cast<double>(p00) * (1.0 - x_frac) * (1.0 - y_frac) +
                       static_cast<double>(p01) * x_frac * (1.0 - y_frac) +
                       static_cast<double>(p10) * (1.0 - x_frac) * y_frac +
                       static_cast<double>(p11) * x_frac * y_frac;

        output[idx] = static_cast<float>(value);
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
    double scale_factor,
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
}
