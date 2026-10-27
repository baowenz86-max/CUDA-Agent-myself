// kernels/custom_ops.cu
// Elementwise kernels for clamp and add_scalar, float32 CUDA only.

#include <cuda_runtime.h>

namespace {

// Helper to map linear index to 4D logical indices using strides.
__device__ __forceinline__ void unravel_index(
    int64_t linear,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t &i0, int64_t &i1, int64_t &i2, int64_t &i3)
{
    i0 = linear / (dim1 * dim2 * dim3);
    int64_t rem = linear - i0 * dim1 * dim2 * dim3;
    i1 = rem / (dim2 * dim3);
    rem -= i1 * dim2 * dim3;
    i2 = rem / dim3;
    i3 = rem - i2 * dim3;
    // Stride mapping is handled by caller using these indices
}

// Clamp kernel: output = min(max(input, min_val), max_val), respecting strides.
__global__ void clamp_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float min_val,
    float max_val,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = blockDim.x * gridDim.x;
    for (; idx < numel; idx += step) {
        // Compute logical indices
        int64_t i0 = idx / (dim1 * dim2 * dim3);
        int64_t rem = idx - i0 * dim1 * dim2 * dim3;
        int64_t i1 = rem / (dim2 * dim3);
        rem -= i1 * dim2 * dim3;
        int64_t i2 = rem / dim3;
        int64_t i3 = rem - i2 * dim3;

        int64_t offset = i0 * stride0 + i1 * stride1 + i2 * stride2 + i3 * stride3;
        float v = input[offset];
        v = fminf(fmaxf(v, min_val), max_val);
        output[offset] = v;
    }
}

// Add scalar kernel: output = input + scalar, respecting strides.
__global__ void add_scalar_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float scalar,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t numel)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = blockDim.x * gridDim.x;
    for (; idx < numel; idx += step) {
        int64_t i0 = idx / (dim1 * dim2 * dim3);
        int64_t rem = idx - i0 * dim1 * dim2 * dim3;
        int64_t i1 = rem / (dim2 * dim3);
        rem -= i1 * dim2 * dim3;
        int64_t i2 = rem / dim3;
        int64_t i3 = rem - i2 * dim3;

        int64_t offset = i0 * stride0 + i1 * stride1 + i2 * stride2 + i3 * stride3;
        output[offset] = input[offset] + scalar;
    }
}

} // namespace

// C-interface launchers for binding layer. No PyTorch dependencies.

extern "C" void clamp_forward_launcher(
    float* output,
    const float* input,
    float min_val,
    float max_val,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t numel,
    cudaStream_t stream)
{
    const int block = 256;
    int64_t blocks = (numel + block - 1) / block;
    if (blocks > 0x7FFFFFFF) blocks = 0x7FFFFFFF; // grid cap
    clamp_kernel<<<(unsigned int)blocks, block, 0, stream>>>(
        output, input, min_val, max_val,
        dim0, dim1, dim2, dim3,
        stride0, stride1, stride2, stride3,
        numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // The binding layer will check and raise if needed.
    }
}

extern "C" void add_scalar_forward_launcher(
    float* output,
    const float* input,
    float scalar,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t numel,
    cudaStream_t stream)
{
    const int block = 256;
    int64_t blocks = (numel + block - 1) / block;
    if (blocks > 0x7FFFFFFF) blocks = 0x7FFFFFFF;
    add_scalar_kernel<<<(unsigned int)blocks, block, 0, stream>>>(
        output, input, scalar,
        dim0, dim1, dim2, dim3,
        stride0, stride1, stride2, stride3,
        numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // The binding layer will check and raise if needed.
    }
}
