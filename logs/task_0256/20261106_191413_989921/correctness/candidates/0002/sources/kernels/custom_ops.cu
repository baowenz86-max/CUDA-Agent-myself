// custom_ops.cu: CUDA kernels for zero padding, channel log_softmax, and conv_transpose2d.
// Only CUDA kernels and raw-pointer launchers are allowed in this file.
// No torch/ATen/c10/pybind dependencies.

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
        int ow = linear % W_out;
        int tmp = linear / W_out;
        int oh = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        int ih = oh - top;
        int iw = ow - left;
        if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
            const float* in_ptr = input + n * input_stride_n + c * input_stride_c
                                  + ih * input_stride_h + iw * input_stride_w;
            float* out_ptr = output + n * output_stride_n + c * output_stride_c
                             + oh * output_stride_h + ow * output_stride_w;
            *out_ptr = *in_ptr;
        } else {
            float* out_ptr = output + n * output_stride_n + c * output_stride_c
                             + oh * output_stride_h + ow * output_stride_w;
            *out_ptr = 0.0f;
        }
    }
}

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
    if (N <= 0 || C <= 0 || H_in <= 0 || W_in <= 0 || H_out <= 0 || W_out <= 0) {
        return;
    }
    if (output == nullptr || input == nullptr) {
        return;
    }
    int total = N * C * H_out * W_out;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535;
    zero_pad2d_kernel<<<grid, block, 0, stream>>>(
        output, input, N, C, H_in, W_in,
        H_out, W_out, left, top,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        output_stride_n, output_stride_c, output_stride_h, output_stride_w);
}

// -----------------------------------------------------------------------------
// LogSoftmax along channel dimension (dim=1).
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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pos = N * H * W;
    for (int pos = idx; pos < total_pos; pos += blockDim.x * gridDim.x) {
        int w = pos % W;
        int tmp = pos / W;
        int h = tmp % H;
        int n = tmp / H;

        float max_val = -FLT_MAX;
        const float* base_in = input + n * input_stride_n + h * input_stride_h + w * input_stride_w;
        for (int c = 0; c < C; ++c) {
            float val = base_in[c * input_stride_c];
            if (val > max_val) max_val = val;
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < C; ++c) {
            float val = base_in[c * input_stride_c];
            sum_exp += expf(val - max_val);
        }
        float log_sum = logf(sum_exp);

        float* base_out = output + n * output_stride_n + h * output_stride_h + w * output_stride_w;
        for (int c = 0; c < C; ++c) {
            float val = base_in[c * input_stride_c];
            base_out[c * output_stride_c] = val - max_val - log_sum;
        }
    }
}

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
        return;
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

// -----------------------------------------------------------------------------
// ConvTranspose2d kernel: direct convolution for each output element.
// This implements ConvTranspose2d with given stride, padding, output_padding.
// Input shape [N, C_in, H_in, W_in], weight [C_in, C_out, kH, kW], bias [C_out].
// Output shape [N, C_out, H_out, W_out].
// Output coordinates map to input via: ih = (oh + pad_h - kh) / stride_h, but only valid if divisible.
// We iterate over kernel positions kH, kW and input channels.
// Since ConvTranspose2d is equivalent to gradient of Conv2d w.r.t. input,
// we can compute as: output[n, co, oh, ow] = bias[co] + sum_{ci, kh, kw} input[n, ci, ih, iw] * weight[ci, co, kh, kw]
// where ih = (oh + pad_h - kh) / stride_h and iw = (ow + pad_w - kw) / stride_w, with divisibility
// and bounds check. This is efficient enough for this problem size.
// -----------------------------------------------------------------------------
__global__ void conv_transpose2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int output_pad_h, int output_pad_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    for (int linear = idx; linear < total; linear += blockDim.x * gridDim.x) {
        int ow = linear % W_out;
        int tmp = linear / W_out;
        int oh = tmp % H_out;
        tmp /= H_out;
        int co = tmp % C_out;
        int n = tmp / C_out;

        float acc = 0.0f;
        for (int ci = 0; ci < C_in; ++ci) {
            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    int ih = oh + pad_h - kh;
                    int iw = ow + pad_w - kw;
                    if (ih < 0 || iw < 0) continue;
                    if (ih % stride_h != 0 || iw % stride_w != 0) continue;
                    ih /= stride_h;
                    iw /= stride_w;
                    if (ih >= H_in || iw >= W_in) continue;
                    // input index: n, ci, ih, iw  (assuming contiguous NCHW)
                    int in_idx = ((n * C_in + ci) * H_in + ih) * W_in + iw;
                    // weight index: ci, co, kh, kw
                    int w_idx = ((ci * C_out + co) * kH + kh) * kW + kw;
                    acc += input[in_idx] * weight[w_idx];
                }
            }
        }
        if (bias != nullptr) {
            acc += bias[co];
        }
        output[linear] = acc;
    }
}

extern "C" void conv_transpose2d_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C_in, int C_out, int H_in, int W_in, int H_out, int W_out,
    int kH, int kW,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int output_pad_h, int output_pad_w,
    cudaStream_t stream)
{
    if (N <= 0 || C_in <= 0 || C_out <= 0 || H_in <= 0 || W_in <= 0 || H_out <= 0 || W_out <= 0) {
        return;
    }
    if (output == nullptr || input == nullptr || weight == nullptr) {
        return;
    }
    int total = N * C_out * H_out * W_out;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535;
    conv_transpose2d_kernel<<<grid, block, 0, stream>>>(
        output, input, weight, bias,
        N, C_in, C_out, H_in, W_in, H_out, W_out,
        kH, kW, stride_h, stride_w, pad_h, pad_w, output_pad_h, output_pad_w);
}
