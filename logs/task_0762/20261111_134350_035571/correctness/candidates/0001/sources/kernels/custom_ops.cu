// custom_ops.cu - CUDA kernels and launchers for the custom extension.
// This file contains only raw CUDA kernels and C-linkage launchers.
// No PyTorch/ATen/c10 headers or Tensor wrappers.
#include <cuda_runtime.h>
#include <cstdint>

// Constant padding kernel for a 4D tensor (N,C,H,W) to (N,C,H_pad,W_pad).
// Input may have arbitrary strides; output is assumed contiguous (row-major).
// Each thread handles one output (oh, ow) position for a given (n, c).
// Grid-stride loop covers all H_pad*W_pad output elements per plane.
__global__ void constant_pad4d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int64_t pad_left,
    int64_t pad_top,
    int64_t in_stride_n,
    int64_t in_stride_c,
    int64_t in_stride_h,
    int64_t in_stride_w,
    float fill_value,
    int64_t total_hw_out)
{
    // Total output height*width per (n,c) plane.
    // Use 64-bit indexing to avoid overflow for large tensors.
    for (int64_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x;
         linear_idx < total_hw_out;
         linear_idx += (int64_t)gridDim.x * blockDim.x)
    {
        int64_t oh = linear_idx / W_out;
        int64_t ow = linear_idx % W_out;

        int64_t ih = oh - pad_top;
        int64_t iw = ow - pad_left;

        float val = fill_value;
        if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
            int64_t n = blockIdx.y / C;
            int64_t c = blockIdx.y - n * C;
            int64_t src_idx = n * in_stride_n + c * in_stride_c + ih * in_stride_h + iw * in_stride_w;
            val = input[src_idx];
        }

        int64_t n = blockIdx.y / C;
        int64_t c = blockIdx.y - n * C;
        int64_t out_idx = ((n * C + c) * H_out + oh) * W_out + ow;
        output[out_idx] = val;
    }
}

extern "C" void custom_pad3d_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom,
    double value,
    int64_t in_stride_n,
    int64_t in_stride_c,
    int64_t in_stride_h,
    int64_t in_stride_w,
    cudaStream_t stream)
{
    // Launch configuration: each block handles a block of output hw positions,
    // grid.y = N*C to cover all planes, grid.x covers H_out*W_out chunks.
    const int block_size = 256;
    int64_t total_hw_out = H_out * W_out;
    int64_t grid_x = (total_hw_out + block_size - 1) / block_size;
    // Cap grid_x to reasonable maximum (e.g., 2^31-1) while using grid-stride loop.
    if (grid_x > 2147483647LL) grid_x = 2147483647LL;
    dim3 grid((unsigned int)grid_x, (unsigned int)(N * C));
    dim3 block(block_size);

    constant_pad4d_kernel<<<grid, block, 0, stream>>>(
        input,
        output,
        N,
        C,
        H_in,
        W_in,
        H_out,
        W_out,
        pad_left,
        pad_top,
        in_stride_n,
        in_stride_c,
        in_stride_h,
        in_stride_w,
        (float)value,
        total_hw_out);
}
