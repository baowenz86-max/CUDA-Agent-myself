// CUDA kernels for reflection_pad2d and remainder scalar
#include <cuda_runtime.h>

// Kernel for reflection padding 2D
// Input: x with shape [N, C, H, W], arbitrary strides
// Output: y with shape [N, C, H_out, W_out]
// Padding: left, right, top, bottom
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int H, int W,
    int H_out, int W_out,
    long long in_n_stride, long long in_c_stride, long long in_h_stride, long long in_w_stride,
    long long out_n_stride, long long out_c_stride, long long out_h_stride, long long out_w_stride,
    int pad_left, int pad_right, int pad_top, int pad_bottom
) {
    long long total = (long long)N * C * H_out * W_out;
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int w_out = idx % W_out;
        long long tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        int h_in_orig = h_out - pad_top;
        int w_in_orig = w_out - pad_left;

        // Reflection mapping for height
        int h_in;
        if (h_in_orig < 0) {
            // Reflect above top
            h_in = -h_in_orig - 1; // equivalent to abs(h_in_orig) - 1? For pad_top=1, h_in_orig=-1 => h_in=0; -2=>1
        } else if (h_in_orig >= H) {
            int over = h_in_orig - H;
            h_in = H - 1 - over;
            // for H=1024, h_in_orig=1024 => over=0 => h_in=1023; h_in_orig=1025=>over=1=>h_in=1022
        } else {
            h_in = h_in_orig;
        }

        // Reflection mapping for width
        int w_in;
        if (w_in_orig < 0) {
            w_in = -w_in_orig - 1;
        } else if (w_in_orig >= W) {
            int over = w_in_orig - W;
            w_in = W - 1 - over;
        } else {
            w_in = w_in_orig;
        }

        // Compute input and output linear offsets using strides
        long long in_offset = (long long)n * in_n_stride +
                              (long long)c * in_c_stride +
                              (long long)h_in * in_h_stride +
                              (long long)w_in * in_w_stride;
        long long out_offset = (long long)n * out_n_stride +
                               (long long)c * out_c_stride +
                               (long long)h_out * out_h_stride +
                               (long long)w_out * out_w_stride;

        y[out_offset] = x[in_offset];
    }
}

// Kernel for remainder with scalar divisor
__global__ void remainder_scalar_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long numel,
    float divisor
) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < numel; idx += stride) {
        float val = x[idx];
        // torch.remainder semantics: result = x - divisor * floor(x / divisor)
        y[idx] = val - divisor * floorf(val / divisor);
    }
}

// Launcher for reflection_pad2d
// Input pointers and strides must be legitimate CUDA pointers for valid tensors.
// x may be non-contiguous; we pass strides in elements.
extern "C" void reflection_pad2d_forward_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    int H_out, int W_out,
    long long in_n_stride, long long in_c_stride, long long in_h_stride, long long in_w_stride,
    long long out_n_stride, long long out_c_stride, long long out_h_stride, long long out_w_stride,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    cudaStream_t stream
) {
    long long total = (long long)N * C * H_out * W_out;
    if (total == 0) return;

    // Use 256 threads per block; grid covers a portion and grid-stride loop handles rest
    int threads = 256;
    int blocks = (int)min((total + threads - 1) / threads, (long long)65535);
    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        x, y, N, C, H, W, H_out, W_out,
        in_n_stride, in_c_stride, in_h_stride, in_w_stride,
        out_n_stride, out_c_stride, out_h_stride, out_w_stride,
        pad_left, pad_right, pad_top, pad_bottom
    );
}

// Launcher for remainder scalar
extern "C" void remainder_scalar_forward_launcher(
    const float* x,
    float* y,
    long long numel,
    float divisor,
    cudaStream_t stream
) {
    if (numel == 0) return;

    int threads = 256;
    int blocks = (int)min((numel + threads - 1) / threads, (long long)65535);
    remainder_scalar_kernel<<<blocks, threads, 0, stream>>>(x, y, numel, divisor);
}
