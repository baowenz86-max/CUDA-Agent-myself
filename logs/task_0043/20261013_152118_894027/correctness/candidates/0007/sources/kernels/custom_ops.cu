#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// Helper: convert 4D linear index to 4D coordinates given strides and shape
// -----------------------------------------------------------------------------
__device__ __forceinline__ void unravel_index_4d(
    int64_t idx,
    int64_t d0, int64_t d1, int64_t d2, int64_t d3,
    int64_t &i0, int64_t &i1, int64_t &i2, int64_t &i3)
{
    i3 = idx % d3;
    idx /= d3;
    i2 = idx % d2;
    idx /= d2;
    i1 = idx % d1;
    i0 = idx / d1;
}

// -----------------------------------------------------------------------------
// 1. view_as_real: complex64 (float2) [B,H,W] -> float32 [B,H,W,2]
//    The input is stored as complex numbers; each complex number occupies
//    8 bytes: real (float) then imag (float). Since torch.view_as_real just
//    reinterprets the data, the output shape [B,H,W,2] with last dim stride 1
//    corresponds to a contiguous float array where the last dim contains
//    (real, imag). We assume input is contiguous complex64. For non-contiguous
//    complex input, the binding layer should call .contiguous() first.
// -----------------------------------------------------------------------------
template <typename T>
__global__ void view_as_real_kernel(
    const T* __restrict__ input_complex,   // T = float2 for complex64
    float* __restrict__ output_real,        // float array of size B*H*W*2
    int64_t total_complex)                  // B*H*W
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_complex) return;

    // Read complex value as float2 (or two consecutive floats)
    // Using reinterpret_cast to float2 is safe for complex64 layout.
    // For portability, we can use __ldg or simple load; here we use direct load
    // through pointer of float2 type. However, input_complex is of type T (float2).
    // Actually, complex64 is represented as a struct with two floats in PyTorch,
    // but on device side it's a float2. We'll assume T = float2.
    float2 val = *reinterpret_cast<const float2*>(input_complex + idx);

    int64_t out_idx = idx * 2;
    output_real[out_idx] = val.x;   // real
    output_real[out_idx + 1] = val.y; // imag
}

// -----------------------------------------------------------------------------
// 2. permute_bhwc: float32 [B,H,W,2] -> [B,H,2,W]
//    This is transposing dimensions 2 and 3. We implement a general kernel
//    that works with arbitrary input strides (provided as int64 array).
//    Output is written contiguously (row-major) into output tensor of shape
//    [B,H,2,W]. This kernel is generic and does not assume contiguous input.
// -----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void permute_bhwc_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t B, int64_t H, int64_t W,
    int64_t in_stride_b, int64_t in_stride_h, int64_t in_stride_w, int64_t in_stride_c,
    int64_t total_output)  // B*H*2*W
{
    int64_t idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= total_output) return;

    // Compute output coordinates: [b, h, c, w]
    int64_t w = idx % W;
    int64_t tmp = idx / W;
    int64_t c = tmp % 2;
    tmp /= 2;
    int64_t h = tmp % H;
    int64_t b = tmp / H;

    // Input coordinates: [b, h, w, c]
    int64_t src_idx = b * in_stride_b + h * in_stride_h + w * in_stride_w + c * in_stride_c;
    output[idx] = input[src_idx];
}

// -----------------------------------------------------------------------------
// 3. clip: float32 [B,H,2,W] (or any shape) elementwise clamp
//    Input may be non-contiguous; we pass strides. Output is contiguous.
//    We use a 1D grid over the total number of elements, unravel 4D.
// -----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void clip_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float min_val, float max_val,
    int64_t B, int64_t H, int64_t C, int64_t W,
    int64_t in_stride_b, int64_t in_stride_h, int64_t in_stride_c, int64_t in_stride_w,
    int64_t total) // B*H*C*W
{
    int64_t idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= total) return;

    int64_t i0, i1, i2, i3;
    unravel_index_4d(idx, B, H, C, W, i0, i1, i2, i3);

    int64_t src_idx = i0 * in_stride_b + i1 * in_stride_h + i2 * in_stride_c + i3 * in_stride_w;
    float val = input[src_idx];
    val = fminf(fmaxf(val, min_val), max_val);
    output[idx] = val;
}

// -----------------------------------------------------------------------------
// Launcher functions (extern "C" to be called from binding.cpp)
// -----------------------------------------------------------------------------

// Launcher for view_as_real.
// Input: pointer to complex64 data (float2 array) on device.
// Output: pointer to float32 array on device (size B*H*W*2).
// total_complex = B*H*W
// stream: CUDA stream to use.
extern "C" void view_as_real_launcher(
    const void* input_complex,   // will be cast to const float2*
    float* output_real,
    int64_t total_complex,
    cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = 256;
    int64_t blocks = (total_complex + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Grid cap protection: use grid-stride loop if blocks > max grid dim (2^31-1).
    // But for typical sizes it's fine. We'll use a single launch; if total_complex
    // is huge, we need a loop. We'll implement a grid-stride version to be safe.
    // However, the kernel above is not grid-stride. Let's create a grid-stride variant
    // inline or use the same kernel with a loop? Easier: use a loop inside a
    // wrapper kernel. We'll define a grid-stride kernel instead.
    // Since total_complex = 64*2048*2048 = 268,435,456, blocks = 1,048,576 which is
    // within limits (max grid x dim = 2,147,483,647). So fine.
    view_as_real_kernel<float2><<<blocks, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<const float2*>(input_complex),
        output_real,
        total_complex);
}

// Launcher for permute_bhwc.
// Input: float32 array with arbitrary strides (provided).
// Output: contiguous float32 array of shape [B,H,2,W].
// Dimensions are given as int64.
// Strides are in elements, not bytes.
extern "C" void permute_bhwc_launcher(
    const float* input,
    float* output,
    int64_t B, int64_t H, int64_t W,
    int64_t in_stride_b, int64_t in_stride_h, int64_t in_stride_w, int64_t in_stride_c,
    cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = 256;
    int64_t total_output = B * H * 2 * W;
    int64_t blocks = (total_output + BLOCK_SIZE - 1) / BLOCK_SIZE;
    permute_bhwc_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
        input, output,
        B, H, W,
        in_stride_b, in_stride_h, in_stride_w, in_stride_c,
        total_output);
}

// Launcher for clip.
// Input: float32 array with arbitrary strides.
// Output: contiguous float32 array of same shape [B,H,C,W], where C=2 in this task.
// min_val, max_val are floats.extern "C" void clip_launcher(
    const float* input,
    float* output,
    float min_val,
    float max_val,
    int64_t B, int64_t H, int64_t C, int64_t W,
    int64_t in_stride_b, int64_t in_stride_h, int64_t in_stride_c, int64_t in_stride_w,
    cudaStream_t stream)
{
    constexpr int BLOCK_SIZE = 256;
    int64_t total = B * H * C * W;
    int64_t blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    clip_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
        input, output,
        min_val, max_val,
        B, H, C, W,
        in_stride_b, in_stride_h, in_stride_c, in_stride_w,
        total);
}
