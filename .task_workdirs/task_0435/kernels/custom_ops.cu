// PixelUnshuffle CUDA kernel and launcher
// Implements torch.nn.PixelUnshuffle forward pass
// Input: float32 tensor with shape [N, C, H, W] (contiguous)
// Output: float32 tensor with shape [N, C*downscale_factor^2, H/downscale_factor, W/downscale_factor]
// The output value at (n, c*r^2 + i*r + j, h_out, w_out) equals input value at (n, c, h_out*r + i, w_out*r + j)

#include <cuda_runtime.h>

// Kernel: each thread writes one output element
// Uses grid-stride loop to cover arbitrary number of elements
__global__ void pixel_unshuffle_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W, int r,
    long long total_elements)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    int W_out = W / r;
    int H_out = H / r;
    int C_out = C * r * r;

    for (; idx < total_elements; idx += stride) {
        // Compute output coordinates (row-major)
        int w_out = idx % W_out;
        long long tmp = idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c_out = tmp % C_out;
        int n = tmp / C_out;

        // From c_out compute original c, i, j
        int c = c_out / (r * r);
        int ij = c_out % (r * r);
        int i = ij / r;
        int j = ij % r;

        // Input coordinates
        int h_in = h_out * r + i;
        int w_in = w_out * r + j;

        // Input index (input is contiguous, so no strides needed)
        long long input_idx = ((long long)n * C + c) * H * W + (long long)h_in * W + w_in;
        output[idx] = input[input_idx];
    }
}

// C-interface launcher (no PyTorch dependencies)
extern "C" void pixel_unshuffle_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W, int r,
    long long total_elements,
    cudaStream_t stream)
{
    // Choose a reasonable block size and grid size
    int threads = 256;
    long long blocks_ll = (total_elements + threads - 1) / threads;
    // Cap blocks to avoid launching too many blocks (grid-stride loop handles rest)
    int max_blocks = 65535; // typical max grid dim for x, but can be larger; keep simple
    int blocks = (int)min(blocks_ll, (long long)max_blocks);

    pixel_unshuffle_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, r, total_elements);
}
