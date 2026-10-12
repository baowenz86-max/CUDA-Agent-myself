#include <cuda_runtime.h>
#include <stdint.h>

// Cast uint8 tensor to float32.
__global__ void cast_float_forward_kernel(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    output[idx] = (float)input[idx];
}

extern "C" void cast_float_launcher(
    const uint8_t* input,
    float* output,
    int total,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cast_float_forward_kernel<<<blocks, threads, 0, stream>>>(input, output, total);
}
