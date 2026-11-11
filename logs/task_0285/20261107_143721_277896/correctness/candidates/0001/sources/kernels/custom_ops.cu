#include <cuda_runtime.h>

// CUDA kernel for pixel_unshuffle
// Input: float* input, arbitrary strides
// Output: float* output, contiguous
// The operation: output[b, c*r*r + i*r + j, h, w] = input[b, c, h*r + i, w*r + j]
// where r = downscale_factor
__global__ void pixel_unshuffle_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W, int r,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    long long total_elements)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (long long linear_idx = idx; linear_idx < total_elements; linear_idx += stride) {
        // Decode linear index into output coordinates: (n, oc, oh, ow)
        int ow = linear_idx % W;
        long long tmp = linear_idx / W;
        int oh = tmp % H;
        tmp /= H;
        int oc = tmp % (C * r * r);
        int n = tmp / (C * r * r);

        // From oc, derive c and (i,j)
        int c = oc / (r * r);
        int rem = oc % (r * r);
        int i = rem / r;
        int j = rem % r;

        // Input coordinates
        int iw = ow * r + j;
        int ih = oh * r + i;

        // Input offset using strides
        long long in_offset = (long long)n * in_stride_n + (long long)c * in_stride_c
                            + (long long)ih * in_stride_h + (long long)iw * in_stride_w;
        long long out_offset = (long long)n * out_stride_n + (long long)oc * out_stride_c
                             + (long long)oh * out_stride_h + (long long)ow * out_stride_w;

        output[out_offset] = input[in_offset];
    }
}

// Launcher function (extern "C" for easy binding)
extern "C" void pixel_unshuffle_custom_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W, int r,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    int out_H = H / r;
    int out_W = W / r;
    int out_C = C * r * r;
    long long total_elements = (long long)N * out_C * out_H * out_W;

    if (total_elements == 0) return;

    // Configure grid and block
    int block_size = 256;
    // Grid-stride: use enough blocks to cover but not excessive; cap at some reasonable number
    long long blocks_needed = (total_elements + block_size - 1) / block_size;
    // Limit grid to avoid excessive launch for small tensors
    int grid_size = (int)min(blocks_needed, (long long)65535);
    // But for very large tensors grid_size can be up to 65535 as well (max grid dim in x for CUDA)
    pixel_unshuffle_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output,
        N, C, H, W, r,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        total_elements);
}
