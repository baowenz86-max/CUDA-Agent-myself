#include <cuda_runtime.h>
#include <math.h>

// -----------------------------------------------------------------------------
// Unfold kernel
// -----------------------------------------------------------------------------
// Input: (B, C, H, W) with strides (sB, sC, sH, sW)
// Output: (B, C * kernel_h * kernel_w, L), where L = out_h * out_w
// For each output element, determine batch, channel block, output spatial index,
// then find corresponding input element using strides and padding/dilation.
// Grid-stride loop to handle large number of elements (18*45*130560 ~ 105M).
__global__ void unfold_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int H, int W,
    long long sB, long long sC, long long sH, long long sW,
    int kernel_h, int kernel_w,
    int dilation_h, int dilation_w,
    int padding_h, int padding_w,
    int stride_h, int stride_w,
    int out_h, int out_w,
    int out_channels, // C * kernel_h * kernel_w
    long long total_out // B * out_channels * out_h * out_w
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < total_out; idx += stride) {
        // Decompose idx
        int l_w = idx % out_w;
        long long tmp = idx / out_w;
        int l_h = tmp % out_h;
        tmp /= out_h;
        int oc = tmp % out_channels;
        int b = tmp / out_channels;

        // map oc to (c, kh, kw)
        int c = oc / (kernel_h * kernel_w);
        int rem = oc % (kernel_h * kernel_w);
        int kh = rem / kernel_w;
        int kw = rem % kernel_w;

        // input coordinates (with padding)
        int in_h = l_h * stride_h - padding_h + kh * dilation_h;
        int in_w = l_w * stride_w - padding_w + kw * dilation_w;

        float val = 0.0f;
        if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
            // compute input index using strides
            long long input_idx = (long long)b * sB + (long long)c * sC + (long long)in_h * sH + (long long)in_w * sW;
            val = input[input_idx];
        }
        output[idx] = val;
    }
}

// Host launcher for unfold (no torch dependencies)
extern "C" void unfold_launcher(
    const float* input,
    float* output,
    int B, int C, int H, int W,
    long long sB, long long sC, long long sH, long long sW,
    int kernel_h, int kernel_w,
    int dilation_h, int dilation_w,
    int padding_h, int padding_w,
    int stride_h, int stride_w,
    cudaStream_t stream
) {
    int out_h = (H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_w = (W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int out_channels = C * kernel_h * kernel_w;
    long long total_out = (long long)B * out_channels * out_h * out_w;

    if (total_out <= 0) return;

    int block_size = 256;
    long long grid_size = (total_out + block_size - 1) / block_size;
    // Cap grid to avoid too many blocks, use grid-stride loop; max grid dim 2^31-1
    if (grid_size > 2147483647LL) grid_size = 2147483647LL;

    unfold_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        input, output, B, C, H, W,
        sB, sC, sH, sW,
        kernel_h, kernel_w, dilation_h, dilation_w,
        padding_h, padding_w, stride_h, stride_w,
        out_h, out_w, out_channels, total_out
    );
}

// -----------------------------------------------------------------------------
// Sign kernel
// -----------------------------------------------------------------------------
__global__ void sign_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long n
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        float val = input[idx];
        output[idx] = (val > 0.0f) - (val < 0.0f); // returns 1, -1, 0 as float
    }
}

extern "C" void sign_launcher(
    const float* input,
    float* output,
    long long n,
    cudaStream_t stream
) {
    if (n <= 0) return;
    int block_size = 256;
    long long grid_size = (n + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL;
    sign_kernel<<<(int)grid_size, block_size, 0, stream>>>(input, output, n);
}

// -----------------------------------------------------------------------------
// Cos kernel
// -----------------------------------------------------------------------------
__global__ void cos_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long n
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        output[idx] = cosf(input[idx]);
    }
}

extern "C" void cos_launcher(
    const float* input,
    float* output,
    long long n,
    cudaStream_t stream
) {
    if (n <= 0) return;
    int block_size = 256;
    long long grid_size = (n + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) grid_size = 2147483647LL;
    cos_kernel<<<(int)grid_size, block_size, 0, stream>>>(input, output, n);
}
