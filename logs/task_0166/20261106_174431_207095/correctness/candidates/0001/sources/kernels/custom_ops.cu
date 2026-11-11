#include <cuda_runtime.h>
#include <math.h>

// Elementwise ReLU kernel: output = max(input, 0)
__global__ void relu_kernel(const float* input, float* output, long long numel) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        float v = input[idx];
        output[idx] = v > 0.0f ? v : 0.0f;
    }
}

// Log-softmax along dim=1.
// input shape: [B, H, W] with arbitrary strides.
// Each thread handles a (b, w) pair and iterates over h to compute max, sum exp, and final values.
// This parallelization avoids cross-thread reduction and works for any strides.
__global__ void log_softmax_dim1_kernel(
    const float* input,
    float* output,
    int B,
    int H,
    int W,
    long long stride_b,
    long long stride_h,
    long long stride_w)
{
    int total_pairs = B * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    int b = idx / W;
    int w = idx % W;

    const float* base = input + (long long)b * stride_b + (long long)w * stride_w;
    float* out_base = output + (long long)b * stride_b + (long long)w * stride_w;

    // Find max over H
    float max_val = -INFINITY;
    for (int h = 0; h < H; ++h) {
        float v = base[(long long)h * stride_h];
        if (v > max_val) max_val = v;
    }

    // Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int h = 0; h < H; ++h) {
        float v = base[(long long)h * stride_h];
        sum += expf(v - max_val);
    }
    float log_sum = logf(sum);

    // Write output
    for (int h = 0; h < H; ++h) {
        float v = base[(long long)h * stride_h];
        out_base[(long long)h * stride_h] = v - max_val - log_sum;
    }
}

// Launcher for ReLU
extern "C" void relu_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

// Launcher for log_softmax along dim=1
extern "C" void log_softmax_dim1_forward_launcher(
    const float* input,
    float* output,
    int B,
    int H,
    int W,
    long long stride_b,
    long long stride_h,
    long long stride_w,
    cudaStream_t stream)
{
    int total_pairs = B * W;
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    if (blocks == 0) return; // no work
    log_softmax_dim1_kernel<<<blocks, threads, 0, stream>>>(
        input, output, B, H, W, stride_b, stride_h, stride_w);
}
