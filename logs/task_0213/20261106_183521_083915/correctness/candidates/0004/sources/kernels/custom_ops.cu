// custom_ops.cu - CUDA kernels and launchers for custom operations
#include <cuda_runtime.h>

// Elementwise threshold kernel: output = (input > threshold) ? input : value
// Uses grid-stride loop to handle arbitrary large tensors with capped grid size.
__global__ void threshold_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    float threshold,
    float value,
    long long total_elements)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < total_elements; idx += stride) {
        float x = input[idx];
        output[idx] = (x > threshold) ? x : value;
    }
}

// Launcher function: grid size capped at 65535, but kernel loops over all elements.
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
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535;
    if (grid_size < 1) grid_size = 1;

    threshold_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, threshold, value, total_elements);
}
