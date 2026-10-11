#include <cuda_runtime.h>

// Unary ceil kernel for float tensors
extern "C" __global__ void ceil_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = ceilf(input[idx]);
    }
}

// Host launcher for ceil
extern "C" void ceil_launcher(
    const float* input,
    float* output,
    int N,
    cudaStream_t stream)
{
    const int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    ceil_kernel<<<gridSize, blockSize, 0, stream>>>(input, output, N);
}
