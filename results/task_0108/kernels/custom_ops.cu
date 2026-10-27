#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

//------------------------------------------------------------------------------
// custom_unfold
//------------------------------------------------------------------------------
extern "C" void custom_unfold_launcher(
    const float* input,
    float* output,
    int64_t batch,
    int64_t in_channels,
    int64_t length,
    int kernel_h,
    int kernel_w,
    int dilation_h,
    int dilation_w,
    int padding_h,
    int padding_w,
    int stride_h,
    int stride_w,
    cudaStream_t stream
);

__global__ void custom_unfold_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t batch,
    int64_t in_channels,
    int64_t length,
    int kernel_h,
    int kernel_w,
    int dilation_h,
    int dilation_w,
    int padding_h,
    int padding_w,
    int stride_h,
    int stride_w,
    int64_t out_h,
    int64_t out_w,
    int64_t L_out,
    int64_t total_output
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_output) return;

    // Decode output indices
    int64_t r = idx / L_out;         // output row = batch * (kernel_h*kernel_w) + kernel_offset
    int64_t c = idx % L_out;         // output column = spatial location within unfolded output

    int64_t b = r / (kernel_h * kernel_w);
    int64_t k_offset = r % (kernel_h * kernel_w);
    int kh = k_offset / kernel_w;
    int kw = k_offset % kernel_w;

    int64_t oh = c / out_w;
    int64_t ow = c % out_w;

    // Compute input indices with padding, dilation, stride
    int64_t in_h = oh * stride_h - padding_h + kh * dilation_h;
    int64_t in_w = ow * stride_w - padding_w + kw * dilation_w;

    float val = 0.0f;
    if (in_h >= 0 && in_h < in_channels && in_w >= 0 && in_w < length) {
        val = input[b * in_channels * length + in_h * length + in_w];
    }
    output[idx] = val;
}

void custom_unfold_launcher(
    const float* input,
    float* output,
    int64_t batch,
    int64_t in_channels,
    int64_t length,
    int kernel_h,
    int kernel_w,
    int dilation_h,
    int dilation_w,
    int padding_h,
    int padding_w,
    int stride_h,
    int stride_w,
    cudaStream_t stream
) {
    // Compute output spatial dimensions based on standard unfold formula.
    // Since input is 3D but treated as 4D with singleton height, we use in_channels as height.
    int64_t out_h = (in_channels + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (length + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int64_t L_out = out_h * out_w;
    int64_t out_batch = batch * kernel_h * kernel_w;
    int64_t total_output = out_batch * L_out;

    int threads = 256;
    int blocks = (total_output + threads - 1) / threads;
    custom_unfold_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch, in_channels, length,
        kernel_h, kernel_w, dilation_h, dilation_w,
        padding_h, padding_w, stride_h, stride_w,
        out_h, out_w, L_out, total_output
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // The binding will check, but we can also throw? Can't throw here, so ignore.
    }
}

//------------------------------------------------------------------------------
// custom_max_pool1d
//------------------------------------------------------------------------------
extern "C" void custom_max_pool1d_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t L,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool ceil_mode,
    cudaStream_t stream
);

__global__ void custom_max_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N,
    int64_t C,
    int64_t L,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool ceil_mode,
    int64_t L_out,
    int64_t total_output
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_output) return;

    int64_t n = idx / (C * L_out);
    int64_t remaining = idx % (C * L_out);
    int64_t c = remaining / L_out;
    int64_t l_out = remaining % L_out;

    float max_val = -FLT_MAX;
    int64_t start = l_out * stride - padding;
    for (int k = 0; k < kernel_size; ++k) {
        int64_t l_in = start + k * dilation;
        if (l_in >= 0 && l_in < L) {
            float val = input[n * C * L + c * L + l_in];
            if (val > max_val) max_val = val;
        }
    }
    output[idx] = max_val;
}

void custom_max_pool1d_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t L,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool ceil_mode,
    cudaStream_t stream
) {
    int64_t L_out;
    if (ceil_mode) {
        L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) / stride + 1;
    } else {
        L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    }
    int64_t total_output = N * C * L_out;

    int threads = 256;
    int blocks = (total_output + threads - 1) / threads;
    custom_max_pool1d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, L,
        kernel_size, stride, padding, dilation, ceil_mode,
        L_out, total_output
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // The binding will check, but we can also throw? Can't throw here, so ignore.
    }
}
