#include <cuda_runtime.h>

#include <cstdint>
#include <cfloat>

// CUDA kernel for 2D max pooling with arbitrary strides, dilation, padding, and ceil_mode
// Supports only float32 as per contract
// Each thread computes one output element; grid-stride loop for robustness
__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t outH, int64_t outW,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int64_t out_stride_n, int64_t out_stride_c, int64_t out_stride_h, int64_t out_stride_w,
    bool ceil_mode
) {
    int64_t total = N * C * outH * outW;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (int64_t)gridDim.x * blockDim.x) {
        // decompose linear index into n, c, oh, ow
        int64_t ow = idx % outW;
        int64_t tmp = idx / outW;
        int64_t oh = tmp % outH;
        tmp /= outH;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // compute input window start in input coordinates
        int64_t h_start = oh * stride_h - pad_h;
        int64_t w_start = ow * stride_w - pad_w;

        // iterate over kernel window (with dilation)
        float max_val = -FLT_MAX;
        bool found = false;

        for (int64_t kh = 0; kh < kernel_h; ++kh) {
            int64_t ih = h_start + kh * dilation_h;
            if (ih < 0 || ih >= H) continue;
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
                int64_t iw = w_start + kw * dilation_w;
                if (iw < 0 || iw >= W) continue;

                // compute input linear offset using provided strides
                int64_t input_idx = n * in_stride_n + c * in_stride_c
                                    + ih * in_stride_h + iw * in_stride_w;
                float val = input[input_idx];
                if (!found || val > max_val) {
                    max_val = val;
                    found = true;
                }
            }
        }

        // If no valid elements (should not happen with padding >= 0 and kernel > 0,
        // but to be safe, output 0 or -FLT_MAX? PyTorch outputs -inf for empty windows?)
        // Actually MaxPool2d with padding and ceil_mode can result in windows that are entirely
        // in padding. In that case PyTorch outputs -inf (or NaN?) For float max pool, it uses
        // -inf as initial value. We'll replicate PyTorch behavior: if no valid element found,
        // output -inf.
        if (!found) {
            max_val = -FLT_MAX; // PyTorch uses -inf (for float, it's -infinity)
            // We use -FLT_MAX, which is close enough? Better use -INFINITY.
            // Let's use -CUDART_INF_F or -INFINITY from math.h
            // But we don't include math.h. We can use -1.0f/0.0f
            max_val = -1.0f / 0.0f; // -inf
        }

        // compute output linear index using provided output strides (output is expected contiguous,
        // but we handle arbitrary strides for completeness)
        int64_t output_idx = n * out_stride_n + c * out_stride_c
                             + oh * out_stride_h + ow * out_stride_w;
        output[output_idx] = max_val;
    }
}

// Host launcher for maxpool2d_forward
// Parameters match binding contract: x, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
// dilation_h, dilation_w, ceil_mode
// All tensor arguments are raw pointers; allocation and checks are in binding.cpp
extern "C" void maxpool2d_forward_launcher(
    const float* input,   // input data pointer (assumed valid for computation, may be non-contiguous)
    float* output,        // output data pointer (contiguous)
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t outH, int64_t outW,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    int64_t dilation_h, int64_t dilation_w,
    bool ceil_mode,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int64_t out_stride_n, int64_t out_stride_c, int64_t out_stride_h, int64_t out_stride_w,
    cudaStream_t stream
) {
    int64_t total = N * C * outH * outW;
    if (total <= 0) return;

    const int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    // Cap to prevent too many blocks (grid-stride loop handles rest)
    int max_grid_dim = 65535; // typical limit for block.x
    if (grid_size > max_grid_dim) grid_size = max_grid_dim;

    dim3 blocks((unsigned int)grid_size);
    dim3 threads(block_size);

    maxpool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, H, W, outH, outW,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
        dilation_h, dilation_w,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        ceil_mode
    );
}
