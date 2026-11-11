#include <cuda_runtime.h>
#include <cmath>

// L2 normalization along dimension 1 (channels) for 4D tensors.
// Input: x shape [B, C, H, W], arbitrary strides.
// Output: y same shape, y[b,c,h,w] = x[b,c,h,w] / max(norm(b,h,w), eps)
// where norm(b,h,w) = sqrt(sum_c x[b,c,h,w]^2 + eps)
__global__ void l2_normalize_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B,
    int C,
    int H,
    int W,
    long long input_stride_b,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    long long output_stride_b,
    long long output_stride_c,
    long long output_stride_h,
    long long output_stride_w,
    float eps)
{
    long long total = (long long)B * H * W;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int b = idx / ((long long)H * W);
        int rem = idx % ((long long)H * W);
        int h = rem / W;
        int w = rem % W;

        // Compute sum of squares over channels
        float sum_sq = 0.0f;
        for (int c = 0; c < C; ++c) {
            long long input_idx = (long long)b * input_stride_b +
                                  (long long)c * input_stride_c +
                                  (long long)h * input_stride_h +
                                  (long long)w * input_stride_w;
            float val = input[input_idx];
            sum_sq += val * val;
        }
        float norm = sqrtf(sum_sq + eps);
        float inv_norm = 1.0f / norm;

        // Write normalized values
        for (int c = 0; c < C; ++c) {
            long long input_idx = (long long)b * input_stride_b +
                                  (long long)c * input_stride_c +
                                  (long long)h * input_stride_h +
                                  (long long)w * input_stride_w;
            long long output_idx = (long long)b * output_stride_b +
                                   (long long)c * output_stride_c +
                                   (long long)h * output_stride_h +
                                   (long long)w * output_stride_w;
            output[output_idx] = input[input_idx] * inv_norm;
        }
    }
}

// Host launcher for L2 normalization along dimension 1.
// This is the only CUDA kernel needed in this stage.
// ConvTranspose will be handled by cuDNN in a separate C++ binding file.
extern "C" void l2_normalize_forward_launcher(
    const float* input,
    float* output,
    int B,
    int C,
    int H,
    int W,
    long long input_stride_b,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    long long output_stride_b,
    long long output_stride_c,
    long long output_stride_h,
    long long output_stride_w,
    float eps,
    cudaStream_t stream)
{
    long long total = (long long)B * H * W;
    if (total == 0) return;

    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    // Limit blocks to a reasonable number for grid-stride loop
    if (blocks > 65535) blocks = 65535;

    l2_normalize_dim1_kernel<<<blocks, threads, 0, stream>>>(
        input, output, B, C, H, W,
        input_stride_b, input_stride_c, input_stride_h, input_stride_w,
        output_stride_b, output_stride_c, output_stride_h, output_stride_w,
        eps);
}
