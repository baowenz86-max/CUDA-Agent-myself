#include <cuda_runtime.h>

// Kernel to transpose dimensions 1 and 2 of a 4D tensor.
// Input shape (N, C, H, W), output shape (N, H, C, W).
// The memory layout for both tensors is contiguous in the transposition order.
// Each thread handles one element in the output tensor based on linear index.
extern "C" __global__ void transpose_1_2_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    // Compute 4D coordinates for output linear index (nhcw order)
    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    tmp = tmp / H;
    int c = tmp % C;
    int n = tmp / C;

    // Input linear index for same element in nchw order
    int src_idx = ((n * C + c) * H + h) * W + w;
    output[idx] = input[src_idx];
}

// Host launcher for transpose_1_2
extern "C" void transpose_1_2_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int total = N * C * H * W;
    const int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    transpose_1_2_kernel<<<gridSize, blockSize, 0, stream>>>(input, output, N, C, H, W);
}
