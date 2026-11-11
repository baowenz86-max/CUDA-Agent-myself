// CUDA kernel for fused relu + cumulative sum along dim=1
// Input: float32 tensor of shape (N, C, H, W)
// Output: same shape, with relu applied then cumulative sum over channels (inclusive)
#include <cuda_runtime.h>

// Every thread computes one output element (n, c, h, w).
// For each spatial location (n, h, w), the cumulative sum over channels is computed sequentially.
// This is a simple but stable implementation; channel dimension 512 is moderately sized.
__global__ void relu_cumsum_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t in_strides_n, int64_t in_strides_c, int64_t in_strides_h, int64_t in_strides_w,
    int64_t out_strides_n, int64_t out_strides_c, int64_t out_strides_h, int64_t out_strides_w)
{
    int64_t total = N * C * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t linear = idx; linear < total; linear += stride) {
        int64_t w = linear % W;
        int64_t tmp = linear / W;
        int64_t h = tmp % H;
        tmp /= H;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // Compute prefix sum over channels for this (n,h,w)
        float sum = 0.0f;
        for (int64_t cc = 0; cc <= c; ++cc) {
            int64_t in_offset = n * in_strides_n + cc * in_strides_c + h * in_strides_h + w * in_strides_w;
            float val = input[in_offset];
            // relu
            val = val > 0.0f ? val : 0.0f;
            sum += val;
        }

        int64_t out_offset = n * out_strides_n + c * out_strides_c + h * out_strides_h + w * out_strides_w;
        output[out_offset] = sum;
    }
}

// Launcher for relu_cumsum_forward.
// Expects input and output tensors on CUDA with float32 dtype.
// Input strides and output strides are arbitrary (may be non-contiguous).
extern "C" void relu_cumsum_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t in_strides_n, int64_t in_strides_c, int64_t in_strides_h, int64_t in_strides_w,
    int64_t out_strides_n, int64_t out_strides_c, int64_t out_strides_h, int64_t out_strides_w,
    cudaStream_t stream)
{
    int64_t total = N * C * H * W;
    const int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    // Cap grid size to avoid exceeding limits; grid-stride loop handles rest
    if (grid_size > 65535) {
        grid_size = 65535;
    }
    relu_cumsum_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input, output, N, C, H, W,
        in_strides_n, in_strides_c, in_strides_h, in_strides_w,
        out_strides_n, out_strides_c, out_strides_h, out_strides_w);
}
