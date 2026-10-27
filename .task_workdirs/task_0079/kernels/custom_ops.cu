#include <cuda_runtime.h>
#include <cstdint>

// ReflectionPad2d kernel for float32 NCHW tensor.
// Input may have arbitrary strides; output is contiguous with shape.
// Padding: left, right, top, bottom.
__global__ void reflection_pad2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    int outH, int outW,
    int pad_left, int pad_top,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * outH * outW) return;

    int w = idx % outW;
    int h = (idx / outW) % outH;
    int c = (idx / (outW * outH)) % C;
    int n = idx / (outW * outH * C);

    // Map output coordinates to input coordinates with reflection.
    int iw = w - pad_left;
    int ih = h - pad_top;

    // Reflect width
    if (iw < 0) iw = -iw;
    else if (iw >= W) iw = 2 * W - 2 - iw;

    // Reflect height
    if (ih < 0) ih = -ih;
    else if (ih >= H) ih = 2 * H - 2 - ih;

    // Clamp to valid range (should not happen with valid padding, but safe)
    iw = max(0, min(iw, W - 1));
    ih = max(0, min(ih, H - 1));

    int64_t in_idx = n * in_stride_n + c * in_stride_c + ih * in_stride_h + iw * in_stride_w;
    output[idx] = input[in_idx];
}

// Element-wise square kernel.
__global__ void square_kernel(float* __restrict__ output, const float* __restrict__ input, int64_t numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        output[i] = val * val;
    }
}

// Launcher: ReflectionPad2d
// Parameters:
//   output: pointer to contiguous output tensor (float)
//   input: pointer to input tensor (float, may be non-contiguous)
//   N, C, H, W: input dims
//   pad_left, pad_right, pad_top, pad_bottom: padding sizes
//   in_stride_n, in_stride_c, in_stride_h, in_stride_w: input strides in elements
//   stream: CUDA stream
extern "C" void pad_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    cudaStream_t stream)
{
    int outH = H + pad_top + pad_bottom;
    int outW = W + pad_left + pad_right;
    int total_elements = N * C * outH * outW;

    const int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    reflection_pad2d_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input,
        N, C, H, W,
        outH, outW,
        pad_left, pad_top,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w);
}

// Launcher: square
//   output: contiguous output tensor
//   input: contiguous input tensor (but kernel allows any)
//   numel: total number of elements
extern "C" void square_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream)
{
    const int block_size = 256;
    // Use grid-stride loop, so grid size can be capped.
    int grid_size = (int)min((int64_t)65535, (numel + block_size - 1) / block_size);
    if (grid_size == 0) grid_size = 1;

    square_kernel<<<grid_size, block_size, 0, stream>>>(output, input, numel);
}
