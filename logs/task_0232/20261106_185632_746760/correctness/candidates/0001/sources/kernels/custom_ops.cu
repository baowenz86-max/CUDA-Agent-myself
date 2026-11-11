#include <cuda_runtime.h>

// Instance normalization forward kernel.
// Each block handles one (n, c) spatial plane of size H*W.
// Block computes mean and variance in shared memory, then normalizes and applies affine.
// Input x is assumed contiguous float32 tensor of shape [N, C, H, W].
// weight and bias are float32 arrays of length C.
__global__ void instance_norm_forward_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C, int HW, float eps)
{
    int nc = blockIdx.x;
    int n = nc / C;
    int c = nc % C;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Base pointer for this (n,c) plane
    const float* x_plane = x + (size_t)nc * HW;
    float* y_plane = y + (size_t)nc * HW;

    // Step 1: compute mean using parallel reduction within block.
    // Since HW may exceed blockSize, use grid-stride within block.
    float sum = 0.0f;
    for (int idx = tid; idx < HW; idx += blockSize) {
        sum += x_plane[idx];
    }

    // Shared memory reduction: array of blockSize floats (or fewer if HW < blockSize)
    extern __shared__ float shared_sum[];
    shared_sum[tid] = sum;
    __syncthreads();

    // Tree reduction
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    float mean = shared_sum[0] / (float)HW;

    // Step 2: compute variance (sum of squared differences)
    float sq_sum = 0.0f;
    for (int idx = tid; idx < HW; idx += blockSize) {
        float diff = x_plane[idx] - mean;
        sq_sum += diff * diff;
    }
    shared_sum[tid] = sq_sum;
    __syncthreads();
    for (int s = blockSize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    float var = shared_sum[0] / (float)HW;

    // Step 3: normalize, apply affine, write output
    float inv_std = rsqrtf(var + eps);
    float w = weight[c];
    float b = bias[c];
    for (int idx = tid; idx < HW; idx += blockSize) {
        y_plane[idx] = (x_plane[idx] - mean) * inv_std * w + b;
    }
}

// Launcher for instance_norm_forward
// Inputs: x, weight, bias are device pointers (float32)
// x shape: N*C*HW contiguous
// weight, bias: length C
// eps: float
// stream: CUDA stream on which to launch
extern "C" void instance_norm_forward_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    int N, int C, int HW, float eps,
    cudaStream_t stream)
{
    if (N <= 0 || C <= 0 || HW <= 0) return;

    // Choose block size: 256 is a good default, but we may need to ensure enough threads for reduction
    int blockSize = 256;
    // For small HW, use smaller block size to avoid wasted threads? Keep simple: use min(256, HW).
    if (HW < blockSize) blockSize = HW;
    // Round up to next power of two for reduction friendly (at least 1)
    int pow2 = 1;
    while (pow2 < blockSize) pow2 <<= 1;
    blockSize = pow2;
    // Cap at 512 for resource usage? 256 is safe. At most 1024.
    if (blockSize > 256) blockSize = 256;

    int gridSize = N * C;

    // Shared memory needed for blockSize floats
    size_t sharedMem = blockSize * sizeof(float);

    instance_norm_forward_kernel<<<gridSize, blockSize, sharedMem, stream>>>(
        x, y, weight, bias, N, C, HW, eps);
}

// Mul scalar kernel: y[i] = x[i] * multiplier
__global__ void mul_scalar_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    float multiplier,
    size_t numel)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        y[idx] = x[idx] * multiplier;
    }
}

// Launcher for mul_scalar
extern "C" void mul_scalar_launcher(
    const float* x,
    float* y,
    float multiplier,
    size_t numel,
    cudaStream_t stream)
{
    if (numel == 0) return;

    int blockSize = 256;
    // Use a reasonable grid size; grid-stride loop handles larger numel.
    int gridSize = (int)((numel + blockSize - 1) / blockSize);
    if (gridSize > 65535) gridSize = 65535; // max grid x dimension

    mul_scalar_kernel<<<gridSize, blockSize, 0, stream>>>(
        x, y, multiplier, numel);
}
