// CUDA kernel for adaptive_avg_pool1d_transpose
// Computes adaptive average pooling over channel dimension after transposing input [B,C,W] to [B,W,C].
// Input is assumed contiguous [B, C, W] as provided by binding after checking/contiguous.
// Output [B, W, output_size].

#include <cuda_runtime.h>

// Kernel: each thread computes one output element (b, w, o)
// For output index o, the pooling interval in input channels is [start, end) computed as:
//   start = floor((o * C) / output_size)
//   end   = ceil(((o + 1) * C) / output_size)
// Average over channels c in [start, end).
// For divisible case when C % output_size == 0, start = o * (C/output_size), end = (o+1)*(C/output_size).
__global__ void adaptive_avg_pool1d_transpose_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int B, int C, int W, int output_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * W * output_size;
    if (idx >= total) return;

    int o = idx % output_size;
    int rem = idx / output_size;
    int w = rem % W;
    int b = rem / W;

    // Input offset for batch b and width w: (b * C + c) * W + w
    // We iterate over c in [start, end)
    int start = (int)(((long long)o * C) / output_size);
    int end = (int)((((long long)(o + 1) * C) + output_size - 1) / output_size);
    if (end > C) end = C;

    double sum = 0.0;
    int count = end - start;
    if (count <= 0) {
        output[idx] = 0.0f;
        return;
    }

    const float* base = input + ((long long)b * C + start) * W + w;
    for (int c = 0; c < count; ++c) {
        sum += base[c * W];
    }
    output[idx] = (float)(sum / count);
}

// Launcher function to be called from binding
void adaptive_avg_pool1d_transpose_launcher(
    float* output,
    const float* input,
    int B, int C, int W, int output_size,
    cudaStream_t stream)
{
    int total = B * W * output_size;
    if (total <= 0) return;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    adaptive_avg_pool1d_transpose_kernel<<<blocks, threads, 0, stream>>>(
        output, input, B, C, W, output_size);
}
