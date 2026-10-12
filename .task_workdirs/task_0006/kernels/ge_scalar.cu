#include <cuda_runtime.h>

__global__ void ge_scalar_kernel(
    const float* __restrict__ input,
    bool* __restrict__ output,
    float scalar,
    long total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (long i = idx; i < total_elements; i += stride) {
        output[i] = (input[i] >= scalar);
    }
}

extern "C" void ge_scalar_launcher(
    const float* input,
    bool* output,
    float scalar,
    long total_elements,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    ge_scalar_kernel<<<blocks, threads, 0, stream>>>(
        input, output, scalar, total_elements);
}
