#include <cuda_runtime.h>

// Reflection padding for 2D: pads last two dims (H, W)
// Input: float32 tensor shape [N, C, H, W]
// Output: float32 tensor shape [N, C, H_out, W_out]
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int w_out = i % W_out;
        int tmp = i / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        // Reflection index calculation
        int h_in = h_out - pad_top;
        if (h_in < 0) h_in = -h_in;
        else if (h_in >= H) h_in = 2 * H - 2 - h_in;

        int w_in = w_out - pad_left;
        if (w_in < 0) w_in = -w_in;
        else if (w_in >= W) w_in = 2 * W - 2 - w_in;

        output[i] = input[((n * C + c) * H + h_in) * W + w_in];
    }
}

// Argsort along dim=1 (channel dimension) for small C (here C=3)
// Each thread sorts one spatial location (n, h, w)
// Input: float tensor shape [N, C, H, W]
// Output: int64 tensor shape [N, C, H, W] containing sorted indices
__global__ void argsort_dim1_kernel(
    const float* __restrict__ input,
    long long* __restrict__ output,
    int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int w = i % W;
        int tmp = i / W;
        int h = tmp % H;
        int n = tmp / H;

        // Load values from channels
        float vals[3]; // assume C=3, but use dynamic indexing if needed
        int inds[3];
        // For general C, we would need dynamic allocation; here we hardcode C=3 for performance
        // But to be safe, handle any C <= 32 with local arrays (max 32 channels)
        // Actually contract says C=3, so we can hardcode, but keep a loop for clarity
        // We'll use a small fixed-size loop up to 8, but C=3
        if (C == 3) {
            vals[0] = input[((n * C + 0) * H + h) * W + w];
            vals[1] = input[((n * C + 1) * H + h) * W + w];
            vals[2] = input[((n * C + 2) * H + h) * W + w];
            inds[0] = 0; inds[1] = 1; inds[2] = 2;
            // Simple insertion sort for 3 elements
            for (int a = 1; a < 3; ++a) {
                float key_val = vals[a];
                int key_idx = inds[a];
                int b = a - 1;
                while (b >= 0 && vals[b] > key_val) {
                    vals[b + 1] = vals[b];
                    inds[b + 1] = inds[b];
                    --b;
                }
                vals[b + 1] = key_val;
                inds[b + 1] = key_idx;
            }
            // Write back sorted indices
            output[((n * C + 0) * H + h) * W + w] = (long long)inds[0];
            output[((n * C + 1) * H + h) * W + w] = (long long)inds[1];
            output[((n * C + 2) * H + h) * W + w] = (long long)inds[2];
        } else {
            // Fallback for arbitrary C (not used in this task, but keep for completeness)
            // Not expected to be called; contract says C=3.
            // To avoid leaving undefined behavior, we can implement a simple loop.
            // However, to keep code simple, we only support C=3 as per contract.
            // If called with other C, it will not write correct results.
            // We can add a check in launcher to assert C==3.
        }
    }
}

// Host launcher for reflection_pad2d
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    cudaStream_t stream) {
    int H_out = H + pad_top + pad_bottom;
    int W_out = W + pad_left + pad_right;
    int total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Cap blocks to avoid grid limit, but use grid-stride loop anyway
    if (blocks > 65535) blocks = 65535;
    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, pad_left, pad_right, pad_top, pad_bottom, H_out, W_out);
}

// Host launcher for argsort_dim1
extern "C" void argsort_dim1_launcher(
    const float* input,
    long long* output,
    int N, int C, int H, int W,
    cudaStream_t stream) {
    // Contract guarantees C=3, but we can assert in binding if needed.
    int total = N * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    argsort_dim1_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W);
}
