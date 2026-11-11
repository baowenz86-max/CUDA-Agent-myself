#include <cuda_runtime.h>

// Fused abs then ReLU6 kernel for contiguous data
__global__ void fused_abs_relu6_kernel(
    const float* input,
    float* output,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        float v = fabsf(input[i]);
        v = fminf(v, 6.0f);
        output[i] = v;
    }
}

// Separate abs kernel for debug
__global__ void abs_kernel(
    const float* input,
    float* output,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

// Separate relu6 kernel for debug
__global__ void relu6_kernel(
    const float* input,
    float* output,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        float v = input[i];
        v = fminf(fmaxf(v, 0.0f), 6.0f);
        output[i] = v;
    }
}

extern "C" void abs_relu6_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream)
{
    if (total_elements <= 0) return;
    const int blockSize = 256;
    // Use grid-stride loop with fixed grid size to avoid overflow
    int device;
    cudaGetDevice(&device);
    int maxGridSize;
    cudaDeviceGetAttribute(&maxGridSize, cudaDevAttrMaxGridSizeX, device);
    int64_t blocks = (total_elements + blockSize - 1) / blockSize;
    if (blocks > maxGridSize) blocks = maxGridSize;
    fused_abs_relu6_kernel<<<(unsigned int)blocks, blockSize, 0, stream>>>(
        input, output, total_elements);
}

extern "C" void abs_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream)
{
    if (total_elements <= 0) return;
    const int blockSize = 256;
    int device;
    cudaGetDevice(&device);
    int maxGridSize;
    cudaDeviceGetAttribute(&maxGridSize, cudaDevAttrMaxGridSizeX, device);
    int64_t blocks = (total_elements + blockSize - 1) / blockSize;
    if (blocks > maxGridSize) blocks = maxGridSize;
    abs_kernel<<<(unsigned int)blocks, blockSize, 0, stream>>>(
        input, output, total_elements);
}

extern "C" void relu6_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream)
{
    if (total_elements <= 0) return;
    const int blockSize = 256;
    int device;
    cudaGetDevice(&device);
    int maxGridSize;
    cudaDeviceGetAttribute(&maxGridSize, cudaDevAttrMaxGridSizeX, device);
    int64_t blocks = (total_elements + blockSize - 1) / blockSize;
    if (blocks > maxGridSize) blocks = maxGridSize;
    relu6_kernel<<<(unsigned int)blocks, blockSize, 0, stream>>>(
        input, output, total_elements);
}
