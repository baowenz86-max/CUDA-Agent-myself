#include <cuda_runtime.h>

// Kernel for L2 normalization along dim=1.
// Input: float tensor of shape [N, C, L]
// Output: same shape, normalized along C (dim=1) for each (n, l).
// p is fixed to 2.0; eps small constant from PyTorch (1e-12) used in F.normalize.
__global__ void normalize_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * L;
    if (idx >= total) return;

    int n = idx / L;
    int l = idx % L;

    const float* in_ptr = input + (size_t)n * C * L + l;
    float sum_sq = 0.0f;

    // Compute sum of squares over channels
    for (int c = 0; c < C; ++c) {
        float val = in_ptr[(size_t)c * L];
        sum_sq += val * val;
    }

    float norm = sqrtf(sum_sq);
    float denom = fmaxf(norm, 1e-12f);
    float inv_denom = 1.0f / denom;

    float* out_ptr = output + (size_t)n * C * L + l;
    for (int c = 0; c < C; ++c) {
        out_ptr[(size_t)c * L] = in_ptr[(size_t)c * L] * inv_denom;
    }
}

// Launcher for normalize op.
// Expects contiguous input and output tensors of shape [N, C, L] float32.
extern "C" void normalize_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int L,
    cudaStream_t stream) {
    int total = N * L;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    normalize_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, L);
}
