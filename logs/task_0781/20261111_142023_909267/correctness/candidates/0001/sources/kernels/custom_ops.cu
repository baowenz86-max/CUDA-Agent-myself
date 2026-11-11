#include <cuda_runtime.h>
#include <cstdint>

// Helper: compute output dimensions for pad operation.
// Input: x is 4D (N, C, H, W), pad is 4 integers (left, right, top, bottom)
// Output: 4D (N, C, H + top + bottom, W + left + right)
__global__ void pad_mean_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t top, int64_t bottom, int64_t left, int64_t right,
    int64_t H_out, int64_t W_out)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * H_out * W_out;
    if (idx >= total) return;

    // Decode output index (n, h_out, w_out)
    int64_t w_out = idx % W_out;
    int64_t tmp = idx / W_out;
    int64_t h_out = tmp % H_out;
    int64_t n = tmp / H_out;

    // Map to input coordinates
    int64_t h_in = h_out - top;
    int64_t w_in = w_out - left;

    // For C==1, output = input if in bounds else 0.
    // More general: sum over C and divide by C, but C is contracted here.
    // Since dim=1 is reduced, we need to compute mean over C, but C=1 so just pick the single channel.
    // We'll implement general C with loop and keepdim false.
    float sum = 0.0f;
    for (int64_t c = 0; c < C; ++c) {
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            int64_t input_idx = ((n * C + c) * H + h_in) * W + w_in;
            sum += input[input_idx];
        }
    }
    output[idx] = sum / static_cast<float>(C);
}

// Kernel for pad only (output 4D)
__global__ void pad_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t top, int64_t bottom, int64_t left, int64_t right,
    int64_t H_out, int64_t W_out)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H_out * W_out;
    if (idx >= total) return;

    int64_t w_out = idx % W_out;
    int64_t tmp = idx / W_out;
    int64_t h_out = tmp % H_out;
    tmp = tmp / H_out;
    int64_t c = tmp % C;
    int64_t n = tmp / C;

    int64_t h_in = h_out - top;
    int64_t w_in = w_out - left;

    float val = 0.0f;
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        int64_t input_idx = ((n * C + c) * H + h_in) * W + w_in;
        val = input[input_idx];
    }
    output[idx] = val;
}

// Kernel for mean over dimension dim=1, with input 4D (N,C,H_out,W_out) -> output 3D (N,H_out,W_out)
// Assumes dim=1 and keepdim=False.
__global__ void mean_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * H * W;
    if (idx >= total) return;

    int64_t w = idx % W;
    int64_t tmp = idx / W;
    int64_t h = tmp % H;
    int64_t n = tmp / H;

    float sum = 0.0f;
    for (int64_t c = 0; c < C; ++c) {
        int64_t input_idx = ((n * C + c) * H + h) * W + w;
        sum += input[input_idx];
    }
    output[idx] = sum / static_cast<float>(C);
}

// Launcher for fused pad_mean: output shape [N, H_out, W_out]
extern "C" void custom_pad_mean_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t top, int64_t bottom, int64_t left, int64_t right,
    cudaStream_t stream)
{
    int64_t H_out = H + top + bottom;
    int64_t W_out = W + left + right;
    int64_t total = N * H_out * W_out;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    pad_mean_kernel<<<grid, block, 0, stream>>>(input, output, N, C, H, W, top, bottom, left, right, H_out, W_out);
}

// Launcher for pad only: output shape [N, C, H_out, W_out]
extern "C" void custom_pad_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t top, int64_t bottom, int64_t left, int64_t right,
    cudaStream_t stream)
{
    int64_t H_out = H + top + bottom;
    int64_t W_out = W + left + right;
    int64_t total = N * C * H_out * W_out;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    pad_kernel<<<grid, block, 0, stream>>>(input, output, N, C, H, W, top, bottom, left, right, H_out, W_out);
}

// Launcher for mean over dim=1: input 4D, output 3D
extern "C" void custom_mean_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    cudaStream_t stream)
{
    int64_t total = N * H * W;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    mean_dim1_kernel<<<grid, block, 0, stream>>>(input, output, N, C, H, W);
}
