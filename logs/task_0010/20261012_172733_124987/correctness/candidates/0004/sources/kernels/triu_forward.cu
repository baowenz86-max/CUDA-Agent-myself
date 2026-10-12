#include <cuda_runtime.h>

// torch.triu on each (H, W) matrix in a batched tensor.
// Input tensor is 4D: (N, C, H, W) float32.
// Zero out elements below the main diagonal (diagonal=0).
__global__ void triu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int h = (idx / W) % H;
    // For each (H,W) matrix, if h <= w (upper triangular including diagonal), keep input; else set 0.
    if (h <= w) {
        output[idx] = input[idx];
    } else {
        output[idx] = 0.0f;
    }
}

extern "C" void triu_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int total = N * C * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    triu_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W);
}
