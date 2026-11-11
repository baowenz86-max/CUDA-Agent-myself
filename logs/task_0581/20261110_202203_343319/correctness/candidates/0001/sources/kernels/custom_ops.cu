#include <cuda_runtime.h>

// Kernel for max over dimension 1 with keepdim=True
// Input: x shape [B, D, W], strides [s0, s1, s2]
// Output: values shape [B, 1, W], indices shape [B, 1, W]
// Each thread computes one output element (b, w)
__global__ void max_dim_keepdim_kernel(
    const float* __restrict__ x,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int B, int D, int W,
    long long s0, long long s1, long long s2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * W;
    if (idx >= total) return;

    int b = idx / W;
    int w = idx % W;

    // Base pointer to x[b, 0, w]
    const float* base = x + b * s0 + w * s2;

    float max_val = -1e30f;
    int max_idx = 0;

    // Loop over D dimension (stride s1)
    for (int d = 0; d < D; ++d) {
        float val = base[d * s1];
        if (val > max_val) {
            max_val = val;
            max_idx = d;
        }
    }

    // Output index: b * W + w
    values[b * W + w] = max_val;
    indices[b * W + w] = (int64_t)max_idx;
}

// Kernel for clone: deep copy preserving data
// Input x shape [N] with stride [s0, s1, ...]? We treat as contiguous 1D for simplicity,
// but contract says input is contiguous [4096,128,512], so we use linear copy.
__global__ void clone_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    long long numel)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long gridSize = blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += gridSize) {
        dst[i] = src[i];
    }
}

// Host launchers with C linkage
extern "C" {

void max_dim_keepdim_launcher(
    const float* x,
    float* values,
    int64_t* indices,
    int B, int D, int W,
    long long s0, long long s1, long long s2,
    cudaStream_t stream)
{
    const int threads = 256;
    int total = B * W;
    int blocks = (total + threads - 1) / threads;
    max_dim_keepdim_kernel<<<blocks, threads, 0, stream>>>(
        x, values, indices, B, D, W, s0, s1, s2);
}

void clone_launcher(
    const float* src,
    float* dst,
    long long numel,
    cudaStream_t stream)
{
    const int threads = 256;
    // grid-stride loop inside kernel allows any grid size, but we set a reasonable number
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // max grid dim x, grid-stride handles rest
    clone_kernel<<<blocks, threads, 0, stream>>>(src, dst, numel);
}

} // extern "C"
