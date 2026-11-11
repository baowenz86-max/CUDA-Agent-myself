#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// custom_pad: ConstantPad2d with value=0
// Input: (N, C, H_in, W_in) with arbitrary strides
// Output: (N, C, H_out, W_out) contiguous, zero-padded
// -----------------------------------------------------------------------------
__global__ void pad_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_output_elements,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out,
    int64_t pad_left, int64_t pad_top,
    int64_t in_N_stride, int64_t in_C_stride, int64_t in_H_stride, int64_t in_W_stride)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < total_output_elements; linear += stride) {
        int64_t ow = linear % W_out;
        int64_t tmp = linear / W_out;
        int64_t oh = tmp % H_out;
        tmp /= H_out;
        int64_t oc = tmp % C;
        int64_t on = tmp / C;

        int64_t iw = ow - pad_left;
        int64_t ih = oh - pad_top;

        if (iw >= 0 && iw < W_in && ih >= 0 && ih < H_in) {
            int64_t input_offset = on * in_N_stride + oc * in_C_stride + ih * in_H_stride + iw * in_W_stride;
            output[linear] = input[input_offset];
        } else {
            output[linear] = 0.0f;
        }
    }
}

extern "C" void custom_pad_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t pad_left, int64_t pad_right, int64_t pad_top, int64_t pad_bottom,
    int64_t in_N_stride, int64_t in_C_stride, int64_t in_H_stride, int64_t in_W_stride,
    cudaStream_t stream)
{
    int64_t H_out = H_in + pad_top + pad_bottom;
    int64_t W_out = W_in + pad_left + pad_right;
    int64_t total_output_elements = N * C * H_out * W_out;

    if (total_output_elements == 0) return;

    const int threads = 256;
    int64_t blocks = (total_output_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    pad_kernel<<<blocks, threads, 0, stream>>>(
        input, output, total_output_elements,
        N, C, H_in, W_in, H_out, W_out,
        pad_left, pad_top,
        in_N_stride, in_C_stride, in_H_stride, in_W_stride);
}

// -----------------------------------------------------------------------------
// custom_cat_dim1: concatenate along dim=1 (channels)
// Inputs a and b have shape (N, C, H, W) with arbitrary strides
// Output shape (N, 2*C, H, W) contiguous
// -----------------------------------------------------------------------------
__global__ void cat_dim1_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ output,
    int64_t total_output_elements,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t a_N_stride, int64_t a_C_stride, int64_t a_H_stride, int64_t a_W_stride,
    int64_t b_N_stride, int64_t b_C_stride, int64_t b_H_stride, int64_t b_W_stride)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < total_output_elements; linear += stride) {
        // output layout: [N, 2*C, H, W]
        int64_t w = linear % W;
        int64_t tmp = linear / W;
        int64_t h = tmp % H;
        tmp /= H;
        int64_t c = tmp % (2 * C);
        int64_t n = tmp / (2 * C);

        if (c < C) {
            int64_t a_offset = n * a_N_stride + c * a_C_stride + h * a_H_stride + w * a_W_stride;
            output[linear] = a[a_offset];
        } else {
            int64_t b_index = c - C;
            int64_t b_offset = n * b_N_stride + b_index * b_C_stride + h * b_H_stride + w * b_W_stride;
            output[linear] = b[b_offset];
        }
    }
}

extern "C" void custom_cat_dim1_launcher(
    const float* a,
    const float* b,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t a_N_stride, int64_t a_C_stride, int64_t a_H_stride, int64_t a_W_stride,
    int64_t b_N_stride, int64_t b_C_stride, int64_t b_H_stride, int64_t b_W_stride,
    cudaStream_t stream)
{
    int64_t total_output_elements = N * (2 * C) * H * W;
    if (total_output_elements == 0) return;

    const int threads = 256;
    int64_t blocks = (total_output_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cat_dim1_kernel<<<blocks, threads, 0, stream>>>(
        a, b, output, total_output_elements, N, C, H, W,
        a_N_stride, a_C_stride, a_H_stride, a_W_stride,
        b_N_stride, b_C_stride, b_H_stride, b_W_stride);
}
