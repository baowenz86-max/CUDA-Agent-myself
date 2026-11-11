#include <cuda_runtime.h>

// Kernel for mean over dim=1 (size 64) with keepdim=True
// Input: x shape [B, C, H, W] (contiguous), output shape [B, 1, H, W]
// Each block computes one output element (b, 0, h, w)
__global__ void mean_dim1_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int B, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * W;
    if (idx >= total) return;

    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    int b = tmp / H;

    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        int in_idx = ((b * C + c) * H + h) * W + w;
        sum += input[in_idx];
    }
    output[idx] = sum / (float)C;
}

// Kernel for mean over dim=2 (size H=256) with keepdim=True
// Input: x shape [B, 1, H, W] (contiguous), output shape [B, 1, 1, W]
// Each block computes one output element (b, 0, 0, w)
__global__ void mean_dim2_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int B, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * W;
    if (idx >= total) return;

    int w = idx % W;
    int b = idx / W;

    float sum = 0.0f;
    for (int h = 0; h < H; ++h) {
        int in_idx = (b * H + h) * W + w;
        sum += input[in_idx];
    }
    output[idx] = sum / (float)H;
}

// Launcher for mean over dim=1
// Expects input contiguous float32 tensor of shape [B, C, H, W]
// Output shape [B, 1, H, W]
extern "C" void mean_dim1_launcher(const float* input,
                                   float* output,
                                   int B, int C, int H, int W,
                                   cudaStream_t stream) {
    int total = B * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mean_dim1_kernel<<<blocks, threads, 0, stream>>>(input, output, B, C, H, W);
}

// Launcher for mean over dim=2
// Expects input contiguous float32 tensor of shape [B, 1, H, W]
// Output shape [B, 1, 1, W]
extern "C" void mean_dim2_launcher(const float* input,
                                   float* output,
                                   int B, int H, int W,
                                   cudaStream_t stream) {
    int total = B * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    mean_dim2_kernel<<<blocks, threads, 0, stream>>>(input, output, B, H, W);
}
