#include <cuda_runtime.h>

// Element-wise greater-than kernel producing bool output.
// Supports arbitrary strides; no broadcasting assumed (same shape).
__global__ void greater_than_kernel(
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    bool* __restrict__ out,
    long long numel,
    long long stride_x1_0, long long stride_x1_1, long long stride_x1_2, long long stride_x1_3,
    long long stride_x2_0, long long stride_x2_1, long long stride_x2_2, long long stride_x2_3,
    long long stride_out_0, long long stride_out_1, long long stride_out_2, long long stride_out_3,
    long long dim1, long long dim2, long long dim3)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        // Compute 4D indices from linear index
        long long i0 = idx / (dim1 * dim2 * dim3);
        long long rem = idx % (dim1 * dim2 * dim3);
        long long i1 = rem / (dim2 * dim3);
        rem = rem % (dim2 * dim3);
        long long i2 = rem / dim3;
        long long i3 = rem % dim3;

        long long off_x1 = i0 * stride_x1_0 + i1 * stride_x1_1 + i2 * stride_x1_2 + i3 * stride_x1_3;
        long long off_x2 = i0 * stride_x2_0 + i1 * stride_x2_1 + i2 * stride_x2_2 + i3 * stride_x2_3;
        long long off_out = i0 * stride_out_0 + i1 * stride_out_1 + i2 * stride_out_2 + i3 * stride_out_3;

        out[off_out] = (x1[off_x1] > x2[off_x2]);
    }
}

// Concatenation along dimension 1: out = [x1 along dim1, x2 along dim1].
// Inputs x1 and x2 have shape [N, C1, H, W] and [N, C2, H, W], output [N, C1+C2, H, W].
__global__ void concatenate_dim1_kernel(
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    float* __restrict__ out,
    long long N, long long C1, long long C2, long long H, long long W,
    long long stride_x1_0, long long stride_x1_1, long long stride_x1_2, long long stride_x1_3,
    long long stride_x2_0, long long stride_x2_1, long long stride_x2_2, long long stride_x2_3,
    long long stride_out_0, long long stride_out_1, long long stride_out_2, long long stride_out_3)
{
    long long total = N * (C1 + C2) * H * W;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        long long i0 = idx / ((C1 + C2) * H * W);
        long long rem = idx % ((C1 + C2) * H * W);
        long long i1 = rem / (H * W);
        rem = rem % (H * W);
        long long i2 = rem / W;
        long long i3 = rem % W;

        if (i1 < C1) {
            // from x1
            long long off_in = i0 * stride_x1_0 + i1 * stride_x1_1 + i2 * stride_x1_2 + i3 * stride_x1_3;
            long long off_out = i0 * stride_out_0 + i1 * stride_out_1 + i2 * stride_out_2 + i3 * stride_out_3;
            out[off_out] = x1[off_in];
        } else {
            // from x2, channel = i1 - C1
            long long c2 = i1 - C1;
            long long off_in = i0 * stride_x2_0 + c2 * stride_x2_1 + i2 * stride_x2_2 + i3 * stride_x2_3;
            long long off_out = i0 * stride_out_0 + i1 * stride_out_1 + i2 * stride_out_2 + i3 * stride_out_3;
            out[off_out] = x2[off_in];
        }
    }
}

// Launcher for greater_than: output bool tensor has same shape as inputs.
// Dimensions: N, C, H, W. Strides provided as long long.
extern "C" void greater_than_launcher(
    const float* x1,
    const float* x2,
    bool* out,
    long long N, long long C, long long H, long long W,
    long long stride_x1_0, long long stride_x1_1, long long stride_x1_2, long long stride_x1_3,
    long long stride_x2_0, long long stride_x2_1, long long stride_x2_2, long long stride_x2_3,
    long long stride_out_0, long long stride_out_1, long long stride_out_2, long long stride_out_3,
    cudaStream_t stream)
{
    long long numel = N * C * H * W;
    if (numel <= 0) return;
    int block = 256;
    // grid-stride loop, launch enough blocks to cover data with cap
    long long grid = (numel + block - 1) / block;
    if (grid > 65535) grid = 65535;
    greater_than_kernel<<<(int)grid, block, 0, stream>>>(
        x1, x2, out, numel,
        stride_x1_0, stride_x1_1, stride_x1_2, stride_x1_3,
        stride_x2_0, stride_x2_1, stride_x2_2, stride_x2_3,
        stride_out_0, stride_out_1, stride_out_2, stride_out_3,
        C, H, W);
}

// Launcher for concatenate_dim1.
extern "C" void concatenate_dim1_launcher(
    const float* x1,
    const float* x2,
    float* out,
    long long N, long long C1, long long C2, long long H, long long W,
    long long stride_x1_0, long long stride_x1_1, long long stride_x1_2, long long stride_x1_3,
    long long stride_x2_0, long long stride_x2_1, long long stride_x2_2, long long stride_x2_3,
    long long stride_out_0, long long stride_out_1, long long stride_out_2, long long stride_out_3,
    cudaStream_t stream)
{
    long long total = N * (C1 + C2) * H * W;
    if (total <= 0) return;
    int block = 256;
    long long grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535;
    concatenate_dim1_kernel<<<(int)grid, block, 0, stream>>>(
        x1, x2, out,
        N, C1, C2, H, W,
        stride_x1_0, stride_x1_1, stride_x1_2, stride_x1_3,
        stride_x2_0, stride_x2_1, stride_x2_2, stride_x2_3,
        stride_out_0, stride_out_1, stride_out_2, stride_out_3);
}
