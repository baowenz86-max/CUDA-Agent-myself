// Custom CUDA kernels for model_new.py
// Only non-convolution ops: reduce_mean_dim1_keepdim and scalar_mul
// All kernels operate on float32 contiguous CUDA tensors
// Launchers accept raw pointers, dimensions, scalar, and cudaStream_t

#include <cuda_runtime.h>
#include <cstdint>

// Kernel: reduce mean along dim=1 with keepdim=True
// Input shape: [N, C, H, W] with C = 3
// Output shape: [N, 1, H, W]
// Each thread handles one or more output elements using grid-stride loop
__global__ void reduce_mean_dim1_keepdim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N,
    int64_t C,
    int64_t HW,
    int64_t total_output_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t out_idx = idx; out_idx < total_output_elements; out_idx += stride) {
        // Decompose output linear index into (n, hw)
        int64_t n = out_idx / HW;
        int64_t hw = out_idx % HW;

        // Compute base offset in input: n * C * HW + hw
        const float* in_base = input + n * C * HW + hw;
        float sum = 0.0f;
        #pragma unroll
        for (int64_t c = 0; c < C; ++c) {
            sum += in_base[c * HW];
        }
        output[out_idx] = sum / static_cast<float>(C);
    }
}

// Kernel: scalar multiplication
// Input shape: any contiguous 1D or multi-dim, total elements N
__global__ void scalar_mul_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_elements,
    float scalar)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_elements; i += stride) {
        output[i] = input[i] * scalar;
    }
}

// Launcher for reduce_mean_dim1_keepdim
// Input: (N, C, H, W) contiguous float32 CUDA tensor
// Output: (N, 1, H, W) allocated by caller, same device
// Assumes C == 3 as per contract, but works for any C >= 1
extern "C" void reduce_mean_dim1_keepdim_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream)
{
    int64_t HW = H * W;
    int64_t total_output_elements = N * HW;

    // Choose block size: 256 threads
    int block_size = 256;
    // Calculate grid size, capped to avoid excessive blocks, using grid-stride loop
    int64_t grid_size = (total_output_elements + block_size - 1) / block_size;
    // Limit grid to something reasonable, e.g., 65535 (max grid dim for x on all modern GPUs)
    if (grid_size > 65535) {
        grid_size = 65535;
    }

    reduce_mean_dim1_keepdim_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input,
        output,
        N,
        C,
        HW,
        total_output_elements);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In production, caller may want to handle error, but here we just record.
        // Binding layer can check cudaGetLastError after launcher if needed.
    }
}

// Launcher for scalar_mul
// Input: contiguous float32 CUDA tensor with total_elements
// Output: contiguous float32 CUDA tensor of same shape, allocated by caller
extern "C" void scalar_mul_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    float scalar,
    cudaStream_t stream)
{
    int block_size = 256;
    int64_t grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 65535) {
        grid_size = 65535;
    }

    scalar_mul_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input,
        output,
        total_elements,
        scalar);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error handling delegated to binding layer if desired
    }
}
