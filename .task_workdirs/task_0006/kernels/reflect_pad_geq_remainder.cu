#include <cuda_runtime.h>
#include <cstdint>

// Kernel: apply reflection padding, then ge>=0, then remainder with 2.
// Input tensor: N, C, H, W, contiguous fp32.
// Padding: left_pad, right_pad, top_pad, bottom_pad.
// Output shape: N, C, H_new, W_new where H_new = H + top_pad + bottom_pad, W_new = W + left_pad + right_pad.
// Output dtype: int64 (torch::kLong).

__global__ void reflect_pad_geq_remainder_kernel(
    const float* __restrict__ input,
    long long* __restrict__ output,
    int N, int C, int H, int W,
    int H_new, int W_new,
    int top_pad, int left_pad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_new * W_new;
    if (idx >= total) return;

    // Decode indices: output layout is N, C, H_new, W_new contiguous.
    int w_out = idx % W_new;
    int h_out = (idx / W_new) % H_new;
    int c = (idx / (W_new * H_new)) % C;
    int n = idx / (W_new * H_new * C);

    // Compute source coordinates after removing padding.
    int h_in = h_out - top_pad;
    int w_in = w_out - left_pad;

    // Reflect coordinates.
    if (h_in < 0) h_in = -h_in;
    else if (h_in >= H) h_in = 2 * H - 2 - h_in;
    if (w_in < 0) w_in = -w_in;
    else if (w_in >= W) w_in = 2 * W - 2 - w_in;

    // Compute input offset.
    long long input_idx = ((long long)n * C + c) * H * W + (long long)h_in * W + w_in;
    float val = input[input_idx];

    // Apply ge and remainder (ge>=0 -> 1, else 0; remainder with 2 is identity for 0/1).
    long long out_val = (val >= 0.0f) ? 1LL : 0LL;
    output[idx] = out_val;
}

extern "C" void reflect_pad_geq_remainder_launcher(
    const float* input,
    long long* output,
    int N, int C, int H, int W,
    int H_new, int W_new,
    int top_pad, int left_pad,
    cudaStream_t stream
) {
    int total = N * C * H_new * W_new;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    reflect_pad_geq_remainder_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, N, C, H, W, H_new, W_new, top_pad, left_pad);
}
