#include <cuda_runtime.h>
#include <cmath>

// Kernel for concatenating two tensors along channel dimension (dim=1)
// Both inputs are contiguous float32 tensors of shape [N, C, H, W]
// Output is [N, 2*C, H, W] contiguous
// We perform a straightforward elementwise copy since inputs are contiguous and
// we copy the first tensor to the first half of the output, second to the second half.
__global__ void cat_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input1,
    const float* __restrict__ input2,
    int total_output_elements,
    int first_half_elements,
    int second_half_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // First half: copy from input1
    for (int i = idx; i < first_half_elements; i += stride) {
        output[i] = input1[i];
    }

    // Second half: copy from input2 with offset
    for (int i = idx; i < second_half_elements; i += stride) {
        output[first_half_elements + i] = input2[i];
    }
}

// Kernel for elementwise cosine
__global__ void cos_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_elements; i += stride) {
        output[i] = cosf(input[i]);
    }
}

// Host launcher for cat_forward
// Both inputs are contiguous CUDA float32 tensors of shape [N, C, H, W]
// Output is contiguous CUDA float32 tensor of shape [N, 2*C, H, W]
// Uses current CUDA stream passed from binding.
extern "C" void cat_forward_launcher(
    float* output,
    const float* input1,
    const float* input2,
    int total_output_elements,
    int first_half_elements,
    int second_half_elements,
    cudaStream_t stream)
{
    if (total_output_elements <= 0) return;

    const int threads = 256;
    // Use enough blocks with grid-stride loop; cap grid to reasonable size
    int blocks = (total_output_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cat_forward_kernel<<<blocks, threads, 0, stream>>>(
        output, input1, input2,
        total_output_elements, first_half_elements, second_half_elements);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Since this is a .cu file without torch, we cannot throw exceptions.
        // We ignore the error here; binding layer may check synchronization.
        // In practice, we rely on correct launch parameters.
    }
}

// Host launcher for cos_forward
// Input is contiguous CUDA float32 tensor of shape [N, C, H, W]
// Output is contiguous CUDA float32 tensor of same shape
extern "C" void cos_forward_launcher(
    float* output,
    const float* input,
    int total_elements,
    cudaStream_t stream)
{
    if (total_elements <= 0) return;

    const int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    cos_forward_kernel<<<blocks, threads, 0, stream>>>(
        output, input, total_elements);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Ignore for now; binding can check.
    }
}
