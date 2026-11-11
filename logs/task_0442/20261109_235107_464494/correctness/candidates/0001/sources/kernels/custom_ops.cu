// CUDA kernels for custom_abs_forward and custom_pad_forward operations.
// This file contains only CUDA kernels and host launchers with raw pointers.
// No PyTorch/Tensor dependencies.

#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// Kernel: abs_forward_kernel
// Computes output[i] = fabsf(input[i]) for all elements.
// Supports arbitrary input strides by using a linear index over the total number
// of elements; assumes input and output have identical shape and strides.
// -----------------------------------------------------------------------------
__global__ void abs_forward_kernel(
    float* output,
    const float* input,
    int64_t numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

// -----------------------------------------------------------------------------
// Kernel: pad_forward_kernel
// Copies input tensor into output tensor with zero padding.
// Output tensor shape: [N, C, H_out, W_out] where
// H_out = H + pad_top + pad_bottom, W_out = W + pad_left + pad_right.
// Input and output are assumed to be in contiguous row-major layout.
// The kernel launches one thread per output element, but grid may be limited;
// uses grid-stride loop to handle all elements.
// -----------------------------------------------------------------------------
__global__ void pad_forward_kernel(
    float* output,
    const float* input,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t H_out,
    int64_t W_out,
    int64_t pad_left,
    int64_t pad_top)
{
    int64_t total = N * C * H_out * W_out;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        // Decompose linear index into n,c,oh,ow
        int64_t ow = i % W_out;
        int64_t tmp = i / W_out;
        int64_t oh = tmp % H_out;
        tmp /= H_out;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        float val = 0.0f;
        if (oh >= pad_top && oh < pad_top + H &&
            ow >= pad_left && ow < pad_left + W) {
            int64_t ih = oh - pad_top;
            int64_t iw = ow - pad_left;
            int64_t in_index = ((n * C + c) * H + ih) * W + iw;
            val = input[in_index];
        }
        output[i] = val;
    }
}

// -----------------------------------------------------------------------------
// Host launcher for custom_abs_forward.
// Parameters:
//   output: pointer to output tensor (same shape as input), must be CUDA memory.
//   input: pointer to input tensor, CUDA memory.
//   numel: total number of elements.
//   stream: CUDA stream on which to launch kernel.
// -----------------------------------------------------------------------------
extern "C" void custom_abs_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream)
{
    int block_size = 256;
    // Compute grid size, capped to a reasonable maximum to avoid excessive grids,
    // then use grid-stride loop inside kernel.
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535; // kernel handles via loop
    if (grid_size < 1) grid_size = 1;

    abs_forward_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        output, input, numel);
}

// -----------------------------------------------------------------------------
// Host launcher for custom_pad_forward.
// Parameters:
//   output: pointer to output tensor, shape [N,C,H_out,W_out], CUDA memory.
//   input: pointer to input tensor, shape [N,C,H,W], CUDA memory.
//   N, C, H, W: input dimensions.
//   pad_left, pad_right, pad_top, pad_bottom: padding amounts (right and bottom are
//     implicitly used to compute output shape; kernel needs only left and top for
//     offset but we pass all for clarity).
//   stream: CUDA stream.
// -----------------------------------------------------------------------------
extern "C" void custom_pad_forward_launcher(
    float* output,
    const float* input,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom,
    cudaStream_t stream)
{
    int64_t H_out = H + pad_top + pad_bottom;
    int64_t W_out = W + pad_left + pad_right;
    int64_t total = N * C * H_out * W_out;

    // Initialize output to zero using cudaMemsetAsync; then kernel fills interior.
    // This is efficient and avoids writing zeros in kernel for padding regions.
    cudaMemsetAsync(output, 0, total * sizeof(float), stream);

    int block_size = 256;
    int64_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    if (grid_size < 1) grid_size = 1;

    pad_forward_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        output, input, N, C, H, W, H_out, W_out, pad_left, pad_top);
}

// -----------------------------------------------------------------------------
// Utility to retrieve last CUDA error after kernel launch; binding code may call
// this if needed. Not required for operation but helpful for debugging.
// -----------------------------------------------------------------------------
extern "C" cudaError_t get_last_cuda_error() {
    return cudaGetLastError();
}
