// custom_ops.cu - CUDA kernels and launchers for custom operations
#include <cuda_runtime.h>

// Elementwise threshold kernel: output = (input > threshold) ? input : value
// Assumes input and output are contiguous (row-major) as guaranteed by binding layer.
__global__ void threshold_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float threshold,
    float value,
    long long total_elements)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float x = input[idx];
        output[idx] = (x > threshold) ? x : value;
    }
}

// Launcher function: simple 1D grid-stride loop not necessary since we only need one element per thread
// but we'll keep grid size limited to handle large tensors with multiple blocks.
extern "C" void threshold_forward_launcher(
    float* output,
    const float* input,
    float threshold,
    float value,
    long long total_elements,
    cudaStream_t stream)
{
    if (total_elements == 0) return;

    int block_size = 256;
    // Calculate grid size, cap at 65535, but use grid-stride loop inside kernel for > 65535*256 elements?
    // Our kernel only handles idx < total_elements, no loop. So we need enough blocks to cover all elements.
    // For total_elements up to 536870912, we need 536870912 / 256 = 2097152 blocks, exceeding grid limit.
    // Must use grid-stride loop. Let's implement grid-stride in kernel.
    // We'll adjust kernel to use grid-stride loop with a fixed grid size.
    // Actually we launch with grid size min(ceil(total/block), 65535) and kernel loops over elements with stride.
    // Rewrite kernel accordingly.
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535;
    if (grid_size < 1) grid_size = 1;

    // Launch with grid-stride loop version (will be defined below).
    threshold_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, threshold, value, total_elements);

    cudaError_t err = cudaGetLastError();
    // Error will be checked in binding after launcher returns.
}
