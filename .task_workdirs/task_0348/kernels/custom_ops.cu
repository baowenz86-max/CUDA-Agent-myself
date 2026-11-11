// CUDA kernels for GroupNorm and Tanh custom operations
#include <cuda_runtime.h>
#include <math.h>

// ----------------------------------------------------------------------------
// GroupNorm kernel: Normalizes each (n, c) independently.
// Since num_groups = C, each group has size 1, so this reduces to per-channel
// spatial normalization (InstanceNorm style).
// We compute mean and variance over H*W elements for each channel.
// Then normalize, apply weight and bias (if provided).
// This kernel launches one block per (n, c) pair, with threads cooperatively
// reducing over the spatial dimension.
// ----------------------------------------------------------------------------

// Helper: block reduction for sum and sum of squares
template <int BLOCK_SIZE>
__device__ void block_reduce_sum_sq(float local_sum, float local_sq, float& total_sum, float& total_sq) {
    __shared__ float s_sum[BLOCK_SIZE];
    __shared__ float s_sq[BLOCK_SIZE];
    int tid = threadIdx.x;
    s_sum[tid] = local_sum;
    s_sq[tid] = local_sq;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq[tid] += s_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        total_sum = s_sum[0];
        total_sq = s_sq[0];
    }
}

// Kernel for group norm: one block per (n, c)
template <int BLOCK_SIZE>
__global__ void group_norm_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C, int H, int W,
    float eps)
{
    int spatial_size = H * W;
    int index = blockIdx.x;  // index = n * C + c
    int n = index / C;
    int c = index - n * C;

    const float* inp = input + (size_t)n * C * spatial_size + (size_t)c * spatial_size;
    float* out = output + (size_t)n * C * spatial_size + (size_t)c * spatial_size;

    float local_sum = 0.0f;
    float local_sq = 0.0f;

    // Compute partial sums over the spatial dimension using grid-stride loop
    for (int i = threadIdx.x; i < spatial_size; i += BLOCK_SIZE) {
        float val = inp[i];
        local_sum += val;
        local_sq += val * val;
    }

    float total_sum, total_sq;
    block_reduce_sum_sq<BLOCK_SIZE>(local_sum, local_sq, total_sum, total_sq);

    if (threadIdx.x == 0) {
        float mean = total_sum / spatial_size;
        float var = total_sq / spatial_size - mean * mean;
        float inv_std = rsqrtf(var + eps);

        // Store mean and inv_std in shared memory for all threads
        __shared__ float s_mean;
        __shared__ float s_inv_std;
        s_mean = mean;
        s_inv_std = inv_std;
        __syncthreads();

        // Apply normalization and affine
        float w = (weight != nullptr) ? weight[c] : 1.0f;
        float b = (bias != nullptr) ? bias[c] : 0.0f;

        for (int i = threadIdx.x; i < spatial_size; i += BLOCK_SIZE) {
            out[i] = (inp[i] - s_mean) * s_inv_std * w + b;
        }
    }
}

// Launcher for custom_group_norm
// Note: We use a fixed block size of 256 for simplicity and performance.
extern "C" void custom_group_norm_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C, int H, int W,
    float eps,
    cudaStream_t stream)
{
    const int BLOCK_SIZE = 256;
    int spatial_size = H * W;
    int total_blocks = N * C;
    dim3 grid(total_blocks);
    dim3 block(BLOCK_SIZE);

    group_norm_kernel<BLOCK_SIZE><<<grid, block, 0, stream>>>(
        output, input, weight, bias, N, C, H, W, eps);
}

// ----------------------------------------------------------------------------
// Tanh kernel: element-wise tanh
// ----------------------------------------------------------------------------
__global__ void tanh_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    size_t numel)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < numel; i += stride) {
        output[i] = tanhf(input[i]);
    }
}

// Launcher for custom_tanh
extern "C" void custom_tanh_launcher(
    float* output,
    const float* input,
    size_t numel,
    cudaStream_t stream)
{
    const int BLOCK_SIZE = 256;
    // Calculate grid size: cap at some maximum to avoid launching too many blocks?
    // We can use a grid-stride loop with a fixed number of blocks.
    // Use enough blocks to cover the data with the loop.
    int device;
    cudaGetDevice(&device);
    int max_blocks_per_sm;
    cudaDeviceGetAttribute(&max_blocks_per_sm, cudaDevAttrMaxBlocksPerMultiprocessor, device);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    // Use a grid size that provides good occupancy but not more than needed.
    // A simple heuristic: use min(sm_count * max_blocks_per_sm, (numel + BLOCK_SIZE - 1) / BLOCK_SIZE)
    size_t needed_blocks = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int max_blocks = sm_count * max_blocks_per_sm;
    int grid_size = (needed_blocks < (size_t)max_blocks) ? (int)needed_blocks : max_blocks;
    if (grid_size <= 0) grid_size = 1;

    tanh_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(output, input, numel);
}
