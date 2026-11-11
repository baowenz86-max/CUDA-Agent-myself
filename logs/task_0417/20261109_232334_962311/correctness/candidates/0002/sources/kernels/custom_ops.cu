#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel for 2D max pooling with indices (PyTorch semantics).
// Each thread computes one output element.
// Input: NCHW float32 contiguous.
// Output: NCHW float32 pooled, NCHW int64 indices (flattened offsets within each channel, i*W + j).
// Ties: PyTorch returns the first maximum in row-major order. We use strict '>' when scanning
// (row-major from top-left) so we keep the earliest occurrence.
__global__ void max_pool2d_with_indices_kernel(
    const float* __restrict__ input,
    float* __restrict__ pooled,
    int64_t* __restrict__ indices,
    int N, int C, int H_in, int W_in,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int H_out, int W_out) {

    int total_elements = N * C * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int elem = idx; elem < total_elements; elem += stride) {
        // Decompose linear index into (n, c, oh, ow)
        int ow = elem % W_out;
        int tmp = elem / W_out;
        int oh = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        // Input window top-left (before padding)
        int h_start = oh * stride_h - padding_h;
        int w_start = ow * stride_w - padding_w;

        float max_val = -3.402823466e+38f; // -FLT_MAX
        int64_t max_idx = -1; // will be set after first in-bounds element

        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_in = h_start + kh;
            if (h_in < 0 || h_in >= H_in) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int w_in = w_start + kw;
                if (w_in < 0 || w_in >= W_in) continue;

                // Input index: ((n*C + c)*H_in + h_in)*W_in + w_in
                int64_t input_linear = (((int64_t)n * C + c) * H_in + h_in) * W_in + w_in;
                float val = input[input_linear];
                // Strict > ensures first occurrence wins ties (row-major scan order)
                if (val > max_val) {
                    max_val = val;
                    max_idx = (int64_t)h_in * W_in + w_in; // flattened within channel
                }
            }
        }

        int64_t output_linear = (int64_t)elem;
        pooled[output_linear] = max_val;
        indices[output_linear] = max_idx;
    }
}

extern "C" void max_pool2d_with_indices_forward_launcher(
    const float* input,
    float* pooled,
    int64_t* indices,
    int N, int C, int H_in, int W_in,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    cudaStream_t stream) {

    int H_out = (H_in + 2 * padding_h - (kernel_h - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * padding_w - (kernel_w - 1) - 1) / stride_w + 1;

    int total_elements = N * C * H_out * W_out;
    if (total_elements <= 0) return;

    // Launch configuration: 256 threads per block, enough blocks to cover all elements.
    // Use grid-stride loop for safety if total exceeds grid capacity.
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    // Cap blocks to a reasonable number (e.g., 65535) but grid-stride loop handles any size anyway;
    // better to launch exactly enough (or limited) and rely on grid-stride.
    // GPU max grid dim in x is usually 2^31-1, but practical limits; we can launch up to 65535*256 = ~16M threads.
    // For our total (16*3*1024*1024=50M), we need more blocks; but grid-stride loop works with any number blocks.
    // We'll launch as many blocks as needed up to some cap; if total > blocks*threads, loop covers rest.
    // Using min(total needed, 65535) is safe, but may reduce parallelism. Instead, we can launch a grid that is
    // at most maxBlocksPerGrid, but for modern GPUs it's 2^31-1. We'll just launch computed blocks.
    // However, to avoid launch failure due to grid dimension limit, cap at 2^20 blocks (1,048,576) and use grid-stride.
    const int max_blocks = 1048576;
    if (blocks > max_blocks) blocks = max_blocks;

    max_pool2d_with_indices_kernel<<<blocks, threads, 0, stream>>>(
        input, pooled, indices,
        N, C, H_in, W_in,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        H_out, W_out);
}
