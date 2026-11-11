// custom_ops.cu - CUDA kernels and launchers for custom operations
#include <cuda_runtime.h>

// Elementwise threshold kernel for contiguous tensors.
// output = (input > threshold) ? input : value
__global__ void threshold_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float threshold,
    float value,
    long long total_elements)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long linear_idx = idx; linear_idx < total_elements; linear_idx += stride) {
        float x = input[linear_idx];
        output[linear_idx] = (x > threshold) ? x : value;
    }
}

// Launcher function for threshold_forward
// Assumes input and output are contiguous
void threshold_forward_launcher(
    float* output,
    const float* input,
    float threshold,
    float value,
    long long total_elements,
    cudaStream_t stream)
{
    // Empty tensor check
    if (total_elements == 0) return;

    // Choose block size and grid size
    int block_size = 256;
    long long num_blocks_ll = (total_elements + block_size - 1) / block_size;
    int grid_size = (int)min(num_blocks_ll, 65535LL);
    if (grid_size <= 0) grid_size = 1;

    // Kernel call
    threshold_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, threshold, value, total_elements);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Leave error for binding to check; no action here
    }
}
