#include <cuda_runtime.h>
#include <stdint.h>

// Bitwise XOR between two uint8 tensors of same shape.
// Input1: (N, C, H, W) uint8 (from triu output cast to uint8)
// Input2: (K, 1, 1) uint8, expanded to (N, C, H, W) via broadcasting along channel dimension.
// Output: (N, C, H, W) uint8.
__global__ void bitwise_xor_forward_kernel(
    const uint8_t* __restrict__ input1,
    const uint8_t* __restrict__ bias, // shape (K,1,1)
    uint8_t* __restrict__ output,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    // Compute channel index c = (idx / (H*W)) % C
    int c = (idx / (H * W)) % C;
    uint8_t b = bias[c]; // bias is contiguous size C
    output[idx] = input1[idx] ^ b;
}

extern "C" void bitwise_xor_launcher(
    const uint8_t* input1,
    const uint8_t* bias,
    uint8_t* output,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bitwise_xor_forward_kernel<<<blocks, threads, 0, stream>>>(
        input1, bias, output, N, C, H, W);
}
