#include <cuda_runtime.h>
#include <cstdint>

// Kernel for bilinear upsampling with scale_factor=2.0
// Input: NCHW float32 tensor, output: NCHW scaled tensor
// Supports arbitrary strides (non-contiguous) by passing strides as arguments
__global__ void upsample_bilinear2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    long long in_n_stride, long long in_c_stride, long long in_h_stride, long long in_w_stride,
    long long out_n_stride, long long out_c_stride, long long out_h_stride, long long out_w_stride,
    float scale_h, float scale_w
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * H_out * W_out;
    if (idx >= total) return;

    // Decode linear index to n,c,h_out,w_out
    int w_out = idx % W_out;
    int64_t tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;

    // Compute source coordinates using align_corners=false formula (PyTorch default)
    float src_h = (h_out + 0.5f) * scale_h - 0.5f;
    float src_w = (w_out + 0.5f) * scale_w - 0.5f;
    src_h = fmaxf(src_h, 0.0f);
    src_w = fmaxf(src_w, 0.0f);

    int h0 = (int)src_h;
    int w0 = (int)src_w;
    int h1 = min(h0 + 1, H_in - 1);
    int w1 = min(w0 + 1, W_in - 1);
    // Ensure h0/w0 within bounds (src_h may be <0)
    h0 = max(h0, 0);
    w0 = max(w0, 0);

    float h_lambda = src_h - h0;
    float w_lambda = src_w - w0;

    // Base pointers for this n,c
    const float* in_nc = input + n * in_n_stride + c * in_c_stride;
    float* out_nc = output + n * out_n_stride + c * out_c_stride;

    // Load four source pixels
    float v00 = in_nc[h0 * in_h_stride + w0 * in_w_stride];
    float v01 = in_nc[h0 * in_h_stride + w1 * in_w_stride];
    float v10 = in_nc[h1 * in_h_stride + w0 * in_w_stride];
    float v11 = in_nc[h1 * in_h_stride + w1 * in_w_stride];

    // Bilinear interpolation
    float top = v00 * (1.0f - w_lambda) + v01 * w_lambda;
    float bottom = v10 * (1.0f - w_lambda) + v11 * w_lambda;
    float value = top * (1.0f - h_lambda) + bottom * h_lambda;

    out_nc[h_out * out_h_stride + w_out * out_w_stride] = value;
}

// Kernel for reflection padding: output size H_out = H_in + pad_top + pad_bottom, W_out = W_in + pad_left + pad_right
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int pad_left, int pad_top,
    long long in_n_stride, long long in_c_stride, long long in_h_stride, long long in_w_stride,
    long long out_n_stride, long long out_c_stride, long long out_h_stride, long long out_w_stride
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int64_t tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;

    // Compute reflection source indices
    int h_in = h_out - pad_top;
    int w_in = w_out - pad_left;

    // Reflect without including edges: if h_in < 0 -> -h_in - 1; if h_in >= H_in -> 2*H_in - h_in - 2
    // and clamp to [0, H_in-1] (only when H_in == 1)
    if (h_in < 0) h_in = -h_in;  // equivalent to -h_in - 1 for negative? Actually PyTorch reflection pad uses: if (h_in < 0) h_in = -h_in - 1; else if (h_in >= H_in) h_in = 2*H_in - h_in - 2;
    // Wait use correct formula:
    // Let's compute carefully
}

// Correct reflection kernel (re-implement with proper formulas)
__global__ void reflection_pad2d_kernel_v2(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int pad_left, int pad_top,
    long long in_n_stride, long long in_c_stride, long long in_h_stride, long long in_w_stride,
    long long out_n_stride, long long out_c_stride, long long out_h_stride, long long out_w_stride
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int64_t tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;

    int h_in = h_out - pad_top;
    int w_in = w_out - pad_left;

    // Apply reflection mapping
    if (h_in < 0) {
        h_in = -h_in - 1;
    } else if (h_in >= H_in) {
        h_in = 2 * H_in - h_in - 2;
    }
    if (w_in < 0) {
        w_in = -w_in - 1;
    } else if (w_in >= W_in) {
        w_in = 2 * W_in - w_in - 2;
    }

    // Ensure within bounds after mapping (for H_in=1 case)
    h_in = min(max(h_in, 0), H_in - 1);
    w_in = min(max(w_in, 0), W_in - 1);

    const float* in_nc = input + n * in_n_stride + c * in_c_stride;
    float* out_nc = output + n * out_n_stride + c * out_c_stride;

    float value = in_nc[h_in * in_h_stride + w_in * in_w_stride];
    out_nc[h_out * out_h_stride + w_out * out_w_stride] = value;
}

// Launcher for upsample_bilinear2d
// scale_factor is float, e.g., 2.0f. Output dims computed as floor(H_in * scale_factor)
// For scale_factor=2.0, H_out = H_in*2 exactly.
extern "C" void upsample_bilinear2d_launcher(
    const float* x, float* out,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    float scale_factor,
    long long in_n_stride, long long in_c_stride, long long in_h_stride, long long in_w_stride,
    long long out_n_stride, long long out_c_stride, long long out_h_stride, long long out_w_stride,
    cudaStream_t stream
) {
    // Check for empty output
    if (N == 0 || C == 0 || H_out == 0 || W_out == 0) return;

    // Compute scale_h and scale_w = 1/scale_factor (since output size = input size * scale_factor)
    // In PyTorch align_corners=false, src = (dst + 0.5) * (1/scale_factor) - 0.5
    float scale_h = 1.0f / scale_factor;
    float scale_w = 1.0f / scale_factor;

    int64_t total = (int64_t)N * C * H_out * W_out;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    upsample_bilinear2d_kernel<<<blocks, threads, 0, stream>>>(
        x, out,
        N, C, H_in, W_in, H_out, W_out,
        in_n_stride, in_c_stride, in_h_stride, in_w_stride,
        out_n_stride, out_c_stride, out_h_stride, out_w_stride,
        scale_h, scale_w
    );
}

// Launcher for reflection_pad2d
// pad_left, pad_right, pad_top, pad_bottom are ints
extern "C" void reflection_pad2d_launcher(
    const float* x, float* out,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    long long in_n_stride, long long in_c_stride, long long in_h_stride, long long in_w_stride,
    long long out_n_stride, long long out_c_stride, long long out_h_stride, long long out_w_stride,
    cudaStream_t stream
) {
    if (N == 0 || C == 0 || H_out == 0 || W_out == 0) return;

    int64_t total = (int64_t)N * C * H_out * W_out;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    reflection_pad2d_kernel_v2<<<blocks, threads, 0, stream>>>(
        x, out,
        N, C, H_in, W_in, H_out, W_out,
        pad_left, pad_top,
        in_n_stride, in_c_stride, in_h_stride, in_w_stride,
        out_n_stride, out_c_stride, out_h_stride, out_w_stride
    );
}
