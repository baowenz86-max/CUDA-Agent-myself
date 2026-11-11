// custom_ops.cu
// CUDA kernels for ReLU and MaxPool2d operations.
#include <cuda_runtime.h>
#include <limits>

// -----------------------------------------------------------------------------
// ReLU kernel: element-wise max(0, x)
// Supports arbitrary strides via linear index and manual offset calculation.
// Each thread handles multiple elements using a grid-stride loop.
// -----------------------------------------------------------------------------
__global__ void relu_forward_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    long long total_elements) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long grid_stride = (long long)blockDim.x * gridDim.x;
    for (; idx < total_elements; idx += grid_stride) {
        float v = in[idx];
        out[idx] = (v > 0.0f) ? v : 0.0f;
    }
}

// -----------------------------------------------------------------------------
// MaxPool2d kernel: general N-dimensional indexing for input with arbitrary strides.
// This kernel is optimized for the common 2x2/stride2 case but supports general
// parameters via a loop over the pooling window.
// grid: number of output elements (N*C*outH*outW)
// block: 256 threads
// -----------------------------------------------------------------------------
template <int KERNEL_H, int KERNEL_W, int STRIDE_H, int STRIDE_W, int PAD_H, int PAD_W, int DILATION_H, int DILATION_W>
__global__ void max_pool2d_kernel_specialized(
    float* __restrict__ out,
    const float* __restrict__ in,
    int N, int C, int H, int W,
    int outH, int outW,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w) {

    long long total_out = (long long)N * C * outH * outW;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long grid_stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total_out; idx += grid_stride) {
        // Decompose linear index into 4D coordinates
        int w = idx % outW;
        long long tmp = idx / outW;
        int h = tmp % outH;
        tmp /= outH;
        int c = tmp % C;
        int n = tmp / C;

        // Compute input window start in input coordinates
        int h_in_start = h * STRIDE_H - PAD_H;
        int w_in_start = w * STRIDE_W - PAD_W;

        float max_val = -std::numeric_limits<float>::infinity();
        bool has_value = false;

        // Loop over kernel positions
        #pragma unroll
        for (int kh = 0; kh < KERNEL_H; ++kh) {
            int h_in = h_in_start + kh * DILATION_H;
            if (h_in < 0 || h_in >= H) continue;
            #pragma unroll
            for (int kw = 0; kw < KERNEL_W; ++kw) {
                int w_in = w_in_start + kw * DILATION_W;
                if (w_in < 0 || w_in >= W) continue;
                long long in_offset = (long long)n * in_stride_n +
                                      (long long)c * in_stride_c +
                                      (long long)h_in * in_stride_h +
                                      (long long)w_in * in_stride_w;
                float v = in[in_offset];
                if (v > max_val) max_val = v;
                has_value = true;
            }
        }

        // If no valid input (should not happen with proper ceil_mode), write -inf
        // but PyTorch semantics would leave it undefined; for safety set 0.
        if (!has_value) max_val = -std::numeric_limits<float>::infinity();

        long long out_offset = (long long)n * out_stride_n +
                               (long long)c * out_stride_c +
                               (long long)h * out_stride_h +
                               (long long)w * out_stride_w;
        out[out_offset] = max_val;
    }
}

// Generic max pool kernel for arbitrary kernel/stride/pad/dilation (runtime values)
// This is less optimized but handles any parameters.
__global__ void max_pool2d_kernel_generic(
    float* __restrict__ out,
    const float* __restrict__ in,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w) {

    long long total_out = (long long)N * C * outH * outW;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long grid_stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total_out; idx += grid_stride) {
        int w = idx % outW;
        long long tmp = idx / outW;
        int h = tmp % outH;
        tmp /= outH;
        int c = tmp % C;
        int n = tmp / C;

        int h_in_start = h * stride_h - pad_h;
        int w_in_start = w * stride_w - pad_w;

        float max_val = -std::numeric_limits<float>::infinity();
        bool has_value = false;

        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_in = h_in_start + kh * dilation_h;
            if (h_in < 0 || h_in >= H) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int w_in = w_in_start + kw * dilation_w;
                if (w_in < 0 || w_in >= W) continue;
                long long in_offset = (long long)n * in_stride_n +
                                      (long long)c * in_stride_c +
                                      (long long)h_in * in_stride_h +
                                      (long long)w_in * in_stride_w;
                float v = in[in_offset];
                if (v > max_val) max_val = v;
                has_value = true;
            }
        }

        if (!has_value) max_val = -std::numeric_limits<float>::infinity();

        long long out_offset = (long long)n * out_stride_n +
                               (long long)c * out_stride_c +
                               (long long)h * out_stride_h +
                               (long long)w * out_stride_w;
        out[out_offset] = max_val;
    }
}

// -----------------------------------------------------------------------------
// Host launchers (extern "C" to be called from binding.cpp)
// -----------------------------------------------------------------------------

// ReLU launcher: takes raw pointer, total elements, stream
extern "C" void relu_forward_launcher(
    float* output,
    const float* input,
    long long total_elements,
    cudaStream_t stream) {

    const int block_size = 256;
    // grid size: enough blocks to cover all elements, but not too high (grid-stride loop handles rest)
    long long num_blocks_ll = (total_elements + block_size - 1) / block_size;
    // Cap grid at 65535 blocks per dimension (CUDA limit for 1D grid in older archs), but modern GPUs support more.
    // Use smaller cap to avoid excessive grid launch overhead. We'll cap at 1048576 (2^20) blocks.
    int grid_size = (int)std::min(num_blocks_ll, 1048576LL);
    relu_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, total_elements);
}

// MaxPool2d launcher: specialized for common 2x2/2 case and generic fallback.
// Inputs: output and input raw pointers, dimensions, strides (as long long),
// kernel/stride/pad/dilation values, stream.
extern "C" void max_pool2d_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int outH, int outW,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    cudaStream_t stream) {

    long long total_out = (long long)N * C * outH * outW;
    const int block_size = 256;
    long long num_blocks_ll = (total_out + block_size - 1) / block_size;
    int grid_size = (int)std::min(num_blocks_ll, 1048576LL);

    // Dispatch to specialized kernels for common cases to improve performance.
    // We only use specialized if parameters exactly match certain patterns.
    // Otherwise use generic kernel.
    if (kernel_h == 2 && kernel_w == 2 &&
        stride_h == 2 && stride_w == 2 &&
        pad_h == 0 && pad_w == 0 &&
        dilation_h == 1 && dilation_w == 1) {
        max_pool2d_kernel_specialized<2, 2, 2, 2, 0, 0, 1, 1><<<grid_size, block_size, 0, stream>>>(
            output, input,
            N, C, H, W,
            outH, outW,
            in_stride_n, in_stride_c, in_stride_h, in_stride_w,
            out_stride_n, out_stride_c, out_stride_h, out_stride_w);
    } else {
        max_pool2d_kernel_generic<<<grid_size, block_size, 0, stream>>>(
            output, input,
            N, C, H, W,
            outH, outW,
            kernel_h, kernel_w,
            stride_h, stride_w,
            pad_h, pad_w,
            dilation_h, dilation_w,
            in_stride_n, in_stride_c, in_stride_h, in_stride_w,
            out_stride_n, out_stride_c, out_stride_h, out_stride_w);
    }
}
