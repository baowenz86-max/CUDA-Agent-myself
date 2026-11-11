// CUDA kernels for instance_norm_2d and relu
// This file contains only raw CUDA kernels and launchers.
// No torch/ATen/c10/pybind dependencies. Tensor handling is in binding.cpp.

#include <cuda_runtime.h>
#include <cmath>

// Helper: get thread linear id
typedef unsigned int uint;

// -----------------------------------------------------------------------------
// InstanceNorm2d forward kernel
// Compute mean and variance per (n,c) and normalize.
// One block per (n,c) pair. Each block has BLOCK_DIM threads.
// For H*W large (262144), each thread handles multiple elements.
// -----------------------------------------------------------------------------
template <int BLOCK_DIM>
__global__ void instance_norm_2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    float eps)
{
    // Block index = n * C + c
    int idx = blockIdx.x;
    int n = idx / C;
    int c = idx % C;

    const int HW = H * W;
    const float scale = 1.0f / (float)HW;

    // Shared memory for mean and variance
    __shared__ float s_mean;
    __shared__ float s_var;

    const float* in_ptr = input + ((size_t)n * C + c) * HW;
    float* out_ptr = output + ((size_t)n * C + c) * HW;

    // First: compute sum and sum of squares using all threads
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    for (int i = threadIdx.x; i < HW; i += BLOCK_DIM) {
        float v = in_ptr[i];
        local_sum += v;
        local_sq_sum += v * v;
    }

    // Block reduction for sum and sq_sum
    __shared__ float s_sum[BLOCK_DIM];
    __shared__ float s_sq_sum[BLOCK_DIM];

    s_sum[threadIdx.x] = local_sum;
    s_sq_sum[threadIdx.x] = local_sq_sum;
    __syncthreads();

    // Reduce within block
    for (int stride = BLOCK_DIM / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sq_sum[threadIdx.x] += s_sq_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum = s_sum[0];
        float sq_sum = s_sq_sum[0];
        float mean = sum * scale;
        float var = sq_sum * scale - mean * mean; // biased variance
        if (var < 0.0f) var = 0.0f; // clamp for numerical stability
        s_mean = mean;
        s_var = var;
    }
    __syncthreads();

    float mean = s_mean;
    float var = s_var;
    float inv_std = rsqrtf(var + eps); // 1 / sqrt(var + eps)

    // Second: normalize all elements
    for (int i = threadIdx.x; i < HW; i += BLOCK_DIM) {
        out_ptr[i] = (in_ptr[i] - mean) * inv_std;
    }
}

// -----------------------------------------------------------------------------
// ReLU kernel: elementwise max(0, x)
// -----------------------------------------------------------------------------
__global__ void relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    size_t total)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; i < total; i += stride) {
        float v = input[i];
        output[i] = v > 0.0f ? v : 0.0f;
    }
}

// -----------------------------------------------------------------------------
// Launchers (extern "C" so binding.cpp can call them)
// -----------------------------------------------------------------------------

extern "C" void instance_norm_2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    float eps,
    cudaStream_t stream)
{
    constexpr int BLOCK_DIM = 256;
    int num_blocks = N * C; // one block per instance-channel pair

    // Ensure num_blocks does not exceed max grid dimension (it's 384, safe)
    if (num_blocks <= 0) return;

    instance_norm_2d_kernel<BLOCK_DIM><<<num_blocks, BLOCK_DIM, 0, stream>>>(
        input, output, N, C, H, W, eps);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In binding.cpp we will throw an exception; here we just ignore or minimal handling
        // Since this is pure C++ without torch, we cannot throw torch exceptions.
        // We mark the error, binding layer can check device sync.
        // For simplicity, do nothing; but to aid debugging, we could printf, but avoid.
    }
}

extern "C" void relu_launcher(
    const float* input,
    float* output,
    size_t total,
    cudaStream_t stream)
{
    if (total == 0) return;

    constexpr int BLOCK_DIM = 256;
    // Calculate grid size (cap to avoid overflow; use 32-bit max grid for simplicity)
    // For large total ~100M, blocks = ceil(total/256) ~ 393216, within limit.
    int blocks = (int)((total + BLOCK_DIM - 1) / BLOCK_DIM);
    // Limit grid to max 65535? Actually modern GPUs support 2^31-1 in x, so okay.

    relu_kernel<<<blocks, BLOCK_DIM, 0, stream>>>(input, output, total);
    cudaError_t err = cudaGetLastError();
    (void)err; // caller should check via cudaDeviceSynchronize or cudaGetLastError
}

// End of custom_ops.cu
