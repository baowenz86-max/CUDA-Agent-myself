// custom_ops.cu: CUDA kernels for zero padding and channel log_softmax.
// Only CUDA kernels and raw-pointer launchers are allowed in this file.
// No torch/ATen/c10/pybind dependencies.
// ConvTranspose2d is intentionally not implemented here; it will be handled
// via cuDNN in the binding layer (third stage).

#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

// -----------------------------------------------------------------------------
// ZeroPad2d kernel: pads (left, right, top, bottom) = (1,1,1,1) zeros
// around H,W dimensions. Input can be non-contiguous; output is contiguous.
// Input shape is assumed [N, C, H_in, W_in], output shape [N, C, H_out, W_out]
// where H_out = H_in + top + bottom, W_out = W_in + left + right.
// -----------------------------------------------------------------------------
__global__ void zero_pad2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int left, int top,
    long long input_stride_n,   // batch stride in elements
    long long input_stride_c,   // channel stride
    long long input_stride_h,   // height stride
    long long input_stride_w,   // width stride (usually 1)
    long long output_stride_n,
    long long output_stride_c,
    long long output_stride_h,
    long long output_stride_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    for (int linear = idx; linear < total; linear += blockDim.x * gridDim.x) {
        // Decode linear index into n,c,oh,ow
        int ow = linear % W_out;
        int tmp = linear / W_out;
        int oh = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        // Determine input coordinates if within valid region
        int ih = oh - top;
        int iw = ow - left;
        if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
            // read from input using strides
            const float* in_ptr = input + n * input_stride_n + c * input_stride_c
                                  + ih * input_stride_h + iw * input_stride_w;
            // write to output using strides
            float* out_ptr = output + n * output_stride_n + c * output_stride_c
                             + oh * output_stride_h + ow * output_stride_w;
            *out_ptr = *in_ptr;
        } else {
            // zero pad
            float* out_ptr = output + n * output_stride_n + c * output_stride_c
                             + oh * output_stride_h + ow * output_stride_w;
            *out_ptr = 0.0f;
        }
    }
}

// Launcher for zero pad. Output tensor must be preallocated contiguous with shape [N,C,H_out,W_out].
// input may be non-contiguous. Strides are in elements.
extern "C" void zero_pad2d_launcher(
    float* output,
    const float* input,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int left, int top,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    long long output_stride_n,
    long long output_stride_c,
    long long output_stride_h,
    long long output_stride_w,
    cudaStream_t stream)
{
    // Sanity checks
    if (N <= 0 || C <= 0 || H_in <= 0 || W_in <= 0 || H_out <= 0 || W_out <= 0) {
        return; // empty tensor, nothing to do
    }
    if (output == nullptr || input == nullptr) {
        // caller should ensure validity; this is just defensive
        return;
    }
    int total = N * C * H_out * W_out;
    int block = 256;
    int grid = (total + block - 1) / block;
    // Cap grid to avoid launching too many blocks; use grid-stride loop.
    if (grid > 65535) grid = 65535;
    zero_pad2d_kernel<<<grid, block, 0, stream>>>(
        output, input, N, C, H_in, W_in,
        H_out, W_out, left, top,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        output_stride_n, output_stride_c, output_stride_h, output_stride_w);
}

// -----------------------------------------------------------------------------
// LogSoftmax along channel dimension (dim=1).
// Input shape [N, C, H, W]. For each (n,h,w), compute
//   y_i = x_i - max_j(x_j) - log(sum_j exp(x_j - max_j(x_j)))
// Output is contiguous, same shape as input. Input may be non-contiguous.
// -----------------------------------------------------------------------------
__global__ void log_softmax_channel_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    long long output_stride_n,
    long long output_stride_c,
    long long output_stride_h,
    long long output_stride_w)
{
    // We assign one thread per (n,h,w) position. Total positions = N*H*W.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pos = N * H * W;
    for (int pos = idx; pos < total_pos; pos += blockDim.x * gridDim.x) {
        int w = pos % W;
        int tmp = pos / W;
        int h = tmp % H;
        int n = tmp / H;

        // Find max across channels
        float max_val = -FLT_MAX;
        const float* base_in = input + n * input_stride_n + h * input_stride_h + w * input_stride_w;
        for (int c = 0; c < C; ++c) {
            float val = base_in[c * input_stride_c];
            if (val > max_val) max_val = val;
        }

        // Compute sum of exp(x - max)
        float sum_exp = 0.0f;
        for (int c = 0; c < C; ++c) {
            float val = base_in[c * input_stride_c];
            sum_exp += expf(val - max_val);
        }
        float log_sum = logf(sum_exp);

        // Write outputs
        float* base_out = output + n * output_stride_n + h * output_stride_h + w * output_stride_w;
        for (int c = 0; c < C; ++c) {
            float val = base_in[c * input_stride_c];
            base_out[c * output_stride_c] = val - max_val - log_sum;
        }
    }
}

// Launcher for channel log_softmax. Output must be preallocated contiguous with shape [N,C,H,W].
extern "C" void log_softmax_channel_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    long long output_stride_n,
    long long output_stride_c,
    long long output_stride_h,
    long long output_stride_w,
    cudaStream_t stream)
{
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        return; // empty
    }
    if (output == nullptr || input == nullptr) {
        return;
    }
    int total_pos = N * H * W;
    int block = 256;
    int grid = (total_pos + block - 1) / block;
    if (grid > 65535) grid = 65535;
    log_softmax_channel_kernel<<<grid, block, 0, stream>>>(
        output, input, N, C, H, W,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        output_stride_n, output_stride_c, output_stride_h, output_stride_w);
}
