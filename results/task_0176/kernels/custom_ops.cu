#include <cuda_runtime.h>
#include <math.h>

// Kernel: fused log_softmax along dim=1 followed by sum along dim=1
// Input: x shape (B, H, W), strides: sx_b, sx_h, sx_w
// Output: out shape (B, W), contiguous, strides: so_b = W, so_w = 1
// Each thread block processes one (b,w) pair and reduces over H.
__global__ void log_softmax_sum_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int H, int W,
    long long sx_b, long long sx_h, long long sx_w,
    int so_b, int so_w) {
    // Use 2D grid: (B, ceil(W/blockDim.x)) or use 1D grid with total = B*W
    // We'll use 1D grid with one thread per (b,w), grid-stride loop if needed.
    int total = B * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int b = i / W;
        int w = i % W;

        // Compute pointer to first element of this (b,w) column
        const float* base = x + b * sx_b + w * sx_w;

        // Find max along H (using H stride sx_h)
        float max_val = base[0];
        for (int h = 1; h < H; ++h) {
            float val = base[h * sx_h];
            if (val > max_val) max_val = val;
        }

        // Compute sum exp(x - max)
        float sum_exp = 0.0f;
        for (int h = 0; h < H; ++h) {
            sum_exp += expf(base[h * sx_h] - max_val);
        }

        float log_sum_exp = logf(sum_exp);
        // log_softmax_i = x_i - max - log_sum_exp
        // sum over i of log_softmax_i = sum(x_i) - H*max - H*log_sum_exp
        float sum_x = 0.0f;
        for (int h = 0; h < H; ++h) {
            sum_x += base[h * sx_h];
        }

        float result = sum_x - H * max_val - H * log_sum_exp;
        out[b * so_b + w * so_w] = result;
    }
}

// Kernel: log_softmax only, output same shape as input (B,H,W), with strides.
// We assume output strides same as input strides for simplicity, but binding may allocate contiguous.
// We'll take output strides as parameters.
__global__ void log_softmax_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int B, int H, int W,
    long long sx_b, long long sx_h, long long sx_w,
    long long sy_b, long long sy_h, long long sy_w) {
    int total = B * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int b = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;

        // Compute max in (b,w) column
        const float* col_base = x + b * sx_b + w * sx_w;
        float max_val = col_base[0];
        for (int hh = 1; hh < H; ++hh) {
            float val = col_base[hh * sx_h];
            if (val > max_val) max_val = val;
        }

        // sum exp(x - max)
        float sum_exp = 0.0f;
        for (int hh = 0; hh < H; ++hh) {
            sum_exp += expf(col_base[hh * sx_h] - max_val);
        }
        float log_sum_exp = logf(sum_exp);

        // compute current value
        float x_val = col_base[h * sx_h];
        y[b * sy_b + h * sy_h + w * sy_w] = x_val - max_val - log_sum_exp;
    }
}

// Kernel: sum along dim=1, output shape (B,W).
__global__ void sum_dim_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int H, int W,
    long long sx_b, long long sx_h, long long sx_w,
    int so_b, int so_w) {
    int total = B * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int b = i / W;
        int w = i % W;

        const float* base = x + b * sx_b + w * sx_w;
        float sum = 0.0f;
        for (int h = 0; h < H; ++h) {
            sum += base[h * sx_h];
        }
        out[b * so_b + w * so_w] = sum;
    }
}

extern "C" void log_softmax_sum_launcher(
    const float* x,
    float* out,
    int B, int H, int W,
    long long sx_b, long long sx_h, long long sx_w,
    int so_b, int so_w,
    cudaStream_t stream) {
    int total = B * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid cap, grid-stride loop handles rest
    log_softmax_sum_kernel<<<blocks, threads, 0, stream>>>(
        x, out, B, H, W, sx_b, sx_h, sx_w, so_b, so_w);
}

extern "C" void log_softmax_launcher(
    const float* x,
    float* y,
    int B, int H, int W,
    long long sx_b, long long sx_h, long long sx_w,
    long long sy_b, long long sy_h, long long sy_w,
    cudaStream_t stream) {
    int total = B * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    log_softmax_kernel<<<blocks, threads, 0, stream>>>(
        x, y, B, H, W, sx_b, sx_h, sx_w, sy_b, sy_h, sy_w);
}

extern "C" void sum_dim_launcher(
    const float* x,
    float* out,
    int B, int H, int W,
    long long sx_b, long long sx_h, long long sx_w,
    int so_b, int so_w,
    cudaStream_t stream) {
    int total = B * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    sum_dim_kernel<<<blocks, threads, 0, stream>>>(
        x, out, B, H, W, sx_b, sx_h, sx_w, so_b, so_w);
}
