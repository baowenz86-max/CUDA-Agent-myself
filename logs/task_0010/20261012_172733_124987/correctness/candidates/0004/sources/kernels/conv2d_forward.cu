#include <cuda_runtime.h>

// Custom Conv2d forward kernel for small filter (3x3) and large batch.
// Each thread computes one output element.
// Input: (N, C, H, W) float32, weight: (K, C, R, S) float32, bias: (K,) float32
// Output: (N, K, H_out, W_out) float32
// Padding is implicit: output size = H - R + 1, W - S + 1 (valid convolution).
// The reference uses nn.Conv2d with default padding=0.
__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int K, int R, int S)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K * (H - R + 1) * (W - S + 1);
    if (idx >= total) return;

    int H_out = H - R + 1;
    int W_out = W - S + 1;
    int out_per_batch_k = H_out * W_out;
    // Decompose linear index to n, k, h_out, w_out
    int n = idx / (K * out_per_batch_k);
    int rem = idx % (K * out_per_batch_k);
    int k = rem / out_per_batch_k;
    int spatial = rem % out_per_batch_k;
    int h_out = spatial / W_out;
    int w_out = spatial % W_out;

    float sum = bias[k];
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int h_in = h_out + r;
                int w_in = w_out + s;
                sum += input[((n * C + c) * H + h_in) * W + w_in] *
                       weight[((k * C + c) * R + r) * S + s];
            }
        }
    }
    output[idx] = sum;
}

extern "C" void conv2d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    cudaStream_t stream)
{
    int H_out = H - R + 1;
    int W_out = W - S + 1;
    int total = N * K * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv2d_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C, H, W, K, R, S);
}
