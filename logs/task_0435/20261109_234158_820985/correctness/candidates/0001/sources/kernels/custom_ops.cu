// PixelUnshuffle CUDA kernel
// Implements torch.nn.PixelUnshuffle forward pass
// Input: float32 tensor with shape [N, C, H, W]
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

    for (; idx < total_elements; idx += stride) {
        // Compute output coordinates
        int w_out = idx % (W / r);
        long long tmp = idx / (W / r);
        int h_out = tmp % (H / r);
        tmp /= (H / r);
        int c_out = tmp % (C * r * r);
        int n = tmp / (C * r * r);

        // From c_out compute original c, i, j
        int c = c_out / (r * r);
        int ij = c_out % (r * r);
        int i = ij / r;
        int j = ij % r;

        // Input coordinates
        int h_in = h_out * r + i;
        int w_in = w_out * r + j;

        // Input index (assuming contiguous input; caller ensures contiguity or uses strides?)
        // Contract says arbitrary stride, so we need to handle strides. But launcher receives only dims and pointer? We'll add stride parameters.
        // For now, assume contiguous as contract includes layout "supports strides". Better add stride params to launcher.
        // We will add strideN, strideC, strideH, strideW to kernel and launcher.
        // However, the binding.cpp is fixed? Actually binding is not provided yet, but we can define launcher signature accordingly.
        // Since we are only writing .cu, we define launcher with the needed parameters and document that binding will pass strides.
    }
}
