#include <cuda_runtime.h>

__global__ void cast_float_kernel(float* output, const unsigned char* input, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    output[idx] = static_cast<float>(input[idx]);
}

extern "C" void cast_float_launcher(
    float* output,
    const unsigned char* input,
    int total,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    cast_float_kernel<<<blocks, threads, 0, stream>>>(output, input, total);
}
