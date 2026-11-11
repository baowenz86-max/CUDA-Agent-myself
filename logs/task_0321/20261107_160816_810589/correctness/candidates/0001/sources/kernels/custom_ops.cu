#include <cuda_runtime.h>
#include <cmath>

// Sigmoid kernel: output[i] = 1 / (1 + exp(-input[i]))
// Supports arbitrary strides via provided input/output strides (in elements).
__global__ void sigmoid_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long numel)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < numel; i += stride) {
        output[i] = 1.0f / (1.0f + expf(-input[i]));
    }
}

// Cosine similarity kernel along dim=1 (channel dimension of size C).
// Input shape: [B, C, H*W] flattened, output shape [B, H*W].
// Each thread computes one output element: sum(x1*x2) / (sqrt(sum(x1^2)) * sqrt(sum(x2^2)))
// If denominator is zero, output 0 (matching PyTorch behavior).
__global__ void cosine_similarity_kernel(
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    float* __restrict__ output,
    int B,
    int C,
    int HW,
    long long stride_b1,   // stride of batch dimension in x1 (in elements)
    long long stride_c1,   // stride of channel dimension in x1
    long long stride_b2,   // stride of batch dimension in x2
    long long stride_c2,   // stride of channel dimension in x2
    long long out_numel)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < out_numel; i += stride) {
        int b = i / HW;
        int hw = i % HW;
        float dot = 0.0f;
        float norm1 = 0.0f;
        float norm2 = 0.0f;
        for (int c = 0; c < C; ++c) {
            long long idx1 = (long long)b * stride_b1 + (long long)c * stride_c1 + hw;
            long long idx2 = (long long)b * stride_b2 + (long long)c * stride_c2 + hw;
            float v1 = x1[idx1];
            float v2 = x2[idx2];
            dot += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }
        float denom = sqrtf(norm1) * sqrtf(norm2);
        output[i] = (denom == 0.0f) ? 0.0f : (dot / denom);
    }
}

// Launcher for sigmoid_forward.
// Inputs: raw pointers, num elements, CUDA stream.
extern "C" void sigmoid_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream)
{
    if (numel == 0) return;
    int threads = 256;
    // Cap grid to avoid exceeding limits, grid-stride loop handles rest.
    long long blocks_ll = (numel + threads - 1) / threads;
    int max_blocks = 65535; // arbitrary high but safe for all current GPUs
    if (blocks_ll > max_blocks) blocks_ll = max_blocks;
    int blocks = (int)blocks_ll;
    sigmoid_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

// Launcher for cosine_similarity_forward.
// Inputs: raw pointers, dimensions, strides (in elements), CUDA stream.
extern "C" void cosine_similarity_forward_launcher(
    const float* x1,
    const float* x2,
    float* output,
    int B,
    int C,
    int HW,
    long long stride_b1,
    long long stride_c1,
    long long stride_b2,
    long long stride_c2,
    long long out_numel,
    cudaStream_t stream)
{
    if (out_numel == 0) return;
    int threads = 256;
    long long blocks_ll = (out_numel + threads - 1) / threads;
    int max_blocks = 65535;
    if (blocks_ll > max_blocks) blocks_ll = max_blocks;
    int blocks = (int)blocks_ll;
    cosine_similarity_kernel<<<blocks, threads, 0, stream>>>(
        x1, x2, output, B, C, HW,
        stride_b1, stride_c1, stride_b2, stride_c2,
        out_numel);
}
