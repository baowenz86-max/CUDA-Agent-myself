#include <cuda_runtime.h>

// Kernel to compute mean over channel dimension (dim=1) with keepdim=True.
// Input: x with shape [B, C, H, W], arbitrary strides.
// Output: out with shape [B, 1, H, W], arbitrary strides.
// Each thread processes one (b, h, w) position using grid-stride loop.
__global__ void mean_dim1_keepdim_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B,
    int C,
    int H,
    int W,
    long long x_stride_b,
    long long x_stride_c,
    long long x_stride_h,
    long long x_stride_w,
    long long out_stride_b,
    long long out_stride_h,
    long long out_stride_w)
{
    long long total = (long long)B * H * W;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int b = idx / ((long long)H * W);
        int rem = idx % ((long long)H * W);
        int h = rem / W;
        int w = rem % W;

        const float* base = x + b * x_stride_b + h * x_stride_h + w * x_stride_w;
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            sum += base[c * x_stride_c];
        }
        float mean = sum / C;

        float* out_ptr = out + b * out_stride_b + h * out_stride_h + w * out_stride_w;
        *out_ptr = mean;
    }
}

// Launcher for mean over dim=1 with keepdim=True.
// Parameters:
//   x: pointer to input tensor (float)
//   out: pointer to output tensor (float)
//   B, C, H, W: dimensions
//   x_strides: array of 4 strides (in elements)
//   out_strides: array of 3 strides for output (B, H, W), since C=1
//   stream: CUDA stream
void mean_dim1_keepdim_launcher(
    const float* x,
    float* out,
    int B,
    int C,
    int H,
    int W,
    const long long* x_strides,
    const long long* out_strides,
    cudaStream_t stream)
{
    int threads = 256;
    long long total = (long long)B * H * W;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    mean_dim1_keepdim_kernel<<<blocks, threads, 0, stream>>>(
        x, out, B, C, H, W,
        x_strides[0], x_strides[1], x_strides[2], x_strides[3],
        out_strides[0], out_strides[1], out_strides[2]);
}

// Kernel to tile dimension 1 by repeats factor.
// The input has shape [B, 1, H, W] and output [B, repeats, H, W].
// We copy data and set output strides to achieve broadcast-like non-contiguous view.
// Since the expansion is along dim1 which is singleton, we can simply copy the input data
// for each repeat index. However, the reference output has stride [524288, 65536, 256, 1]
// which is exactly the same as if output was contiguous with shape [B, 8, H, W] but with
// dim1 stride equal to input dim0 stride (65536) and dim0 stride equal to repeats*65536.
// This corresponds to a non-overlapping repetition, so we can write each output element.
__global__ void tile_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B,
    int repeats,
    int H,
    int W,
    long long x_stride_b,
    long long x_stride_h,
    long long x_stride_w,
    long long out_stride_b,
    long long out_stride_r,
    long long out_stride_h,
    long long out_stride_w)
{
    long long total = (long long)B * repeats * H * W;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        int b = idx / ((long long)repeats * H * W);
        int rem = idx % ((long long)repeats * H * W);
        int r = rem / ((long long)H * W);
        int rem2 = rem % ((long long)H * W);
        int h = rem2 / W;
        int w = rem2 % W;

        // Input element at (b, 0, h, w)
        const float* src = x + b * x_stride_b + h * x_stride_h + w * x_stride_w;
        // Output element at (b, r, h, w)
        float* dst = out + b * out_stride_b + r * out_stride_r + h * out_stride_h + w * out_stride_w;
        *dst = *src;
    }
}

// Launcher for tile along dim=1.
// Parameters:
//   x: pointer to input tensor (float) with shape [B,1,H,W]
//   out: pointer to output tensor (float) with shape [B,repeats,H,W]
//   B, repeats, H, W: dimensions
//   x_strides: array of 4 strides for input (in elements)
//   out_strides: array of 4 strides for output (in elements), as specified by reference
//   stream: CUDA stream
void tile_dim1_launcher(
    const float* x,
    float* out,
    int B,
    int repeats,
    int H,
    int W,
    const long long* x_strides,
    const long long* out_strides,
    cudaStream_t stream)
{
    int threads = 256;
    long long total = (long long)B * repeats * H * W;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;

    tile_dim1_kernel<<<blocks, threads, 0, stream>>>(
        x, out, B, repeats, H, W,
        x_strides[0], x_strides[2], x_strides[3],
        out_strides[0], out_strides[1], out_strides[2], out_strides[3]);
}
