#include <cuda_runtime.h>

// Elementwise clamp kernel: y = min(max(x, min_val), max_val)
// Supports arbitrary strides via flat indexing and loop over elements.
__global__ void clamp_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             float min_val,
                             float max_val,
                             long long numel) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float val = input[i];
        val = val < min_val ? min_val : val;
        val = val > max_val ? max_val : val;
        output[i] = val;
    }
}

// Adaptive average pooling kernel for 4D input (N, C, H, W).
// The input may be non-contiguous; we pass strides (in elements) for each dimension.
// output_size is (H_out, W_out).
// Each thread computes one output element (n, c, oh, ow).
__global__ void adaptive_avg_pool2d_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int N, int C, int H_in, int W_in,
                                           int H_out, int W_out,
                                           long long stride_N, long long stride_C,
                                           long long stride_H, long long stride_W,
                                           long long out_stride_N, long long out_stride_C,
                                           long long out_stride_H, long long out_stride_W) {
    long long total_output = (long long)N * C * H_out * W_out;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long grid_stride = (long long)gridDim.x * blockDim.x;

    for (long long out_linear = idx; out_linear < total_output; out_linear += grid_stride) {
        // Decode output indices (n, c, oh, ow)
        int ow = out_linear % W_out;
        long long tmp = out_linear / W_out;
        int oh = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        tmp /= C;
        int n = tmp; // tmp < N

        // Adaptive window boundaries
        int h_start = (int)floorf((float)oh * H_in / H_out);
        int h_end   = (int)ceilf((float)(oh + 1) * H_in / H_out);
        int w_start = (int)floorf((float)ow * W_in / W_out);
        int w_end   = (int)ceilf((float)(ow + 1) * W_in / W_out);

        // Ensure at least one element
        if (h_end <= h_start) h_end = h_start + 1;
        if (w_end <= w_start) w_end = w_start + 1;

        float sum = 0.0f;
        for (int h = h_start; h < h_end; ++h) {
            for (int w = w_start; w < w_end; ++w) {
                long long input_offset = (long long)n * stride_N
                                        + (long long)c * stride_C
                                        + (long long)h * stride_H
                                        + (long long)w * stride_W;
                sum += input[input_offset];
            }
        }
        float count = (float)((h_end - h_start) * (w_end - w_start));

        long long output_offset = (long long)n * out_stride_N
                                 + (long long)c * out_stride_C
                                 + (long long)oh * out_stride_H
                                 + (long long)ow * out_stride_W;
        output[output_offset] = sum / count;
    }
}

extern "C" void clamp_forward_launcher(const float* input,
                                       float* output,
                                       float min_val,
                                       float max_val,
                                       long long numel,
                                       cudaStream_t stream) {
    if (numel <= 0) return;
    int block_size = 256;
    long long grid_size = (numel + block_size - 1) / block_size;
    // Limit grid to avoid excessive launches
    if (grid_size > 65535) grid_size = 65535;
    clamp_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        input, output, min_val, max_val, numel);
}

extern "C" void adaptive_avg_pool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    long long stride_N, long long stride_C, long long stride_H, long long stride_W,
    long long out_stride_N, long long out_stride_C, long long out_stride_H, long long out_stride_W,
    cudaStream_t stream) {
    long long total_output = (long long)N * C * H_out * W_out;
    if (total_output <= 0) return;
    int block_size = 256;
    long long grid_size = (total_output + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    adaptive_avg_pool2d_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out,
        stride_N, stride_C, stride_H, stride_W,
        out_stride_N, out_stride_C, out_stride_H, out_stride_W);
}
