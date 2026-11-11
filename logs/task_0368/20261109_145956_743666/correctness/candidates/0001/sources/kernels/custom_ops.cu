// embedding_log_softmax_forward CUDA implementation
// Input: indices (int64_t, shape [B, S]), weight (float, shape [V, D])
// Output: embedding_out (float, shape [B, S, D]), log_softmax_out (same shape)
// Log softmax along dim=1 (S dimension)

#include <cuda_runtime.h>
#include <float.h>

// Kernel 1: embedding lookup, writes embedding_out
__global__ void embedding_lookup_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ embedding_out,
    int B, int S, int D)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * S * D;
    // grid-stride loop over all output elements
    for (int64_t idx = tid; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int b = idx / ((int64_t)S * D);
        int rem = idx % ((int64_t)S * D);
        int s = rem / D;
        int d = rem % D;
        int64_t index = indices[(int64_t)b * S + s];
        embedding_out[idx] = weight[index * D + d];
    }
}

// Kernel 2: log softmax over dim=1 (S dimension)
// Each thread block handles one (b,d) pair, or use one thread block per group with parallel reduction.
// For simplicity and performance, we use one block with 256 threads per (b,d) pair, reducing over S up to 1024.
__global__ void log_softmax_kernel(
    const float* __restrict__ embedding_in,
    float* __restrict__ log_softmax_out,
    int B, int S, int D)
{
    // blockIdx.x indexes (b,d) pair
    int bd = blockIdx.x;
    int b = bd / D;
    int d = bd % D;
    if (b >= B || d >= D) return;

    // Base pointer for this (b,d) along S
    const float* row = embedding_in + ((int64_t)b * S + 0) * D + d;
    float* out_row = log_softmax_out + ((int64_t)b * S + 0) * D + d;

    // Step 1: compute max over S
    float max_val = -FLT_MAX;
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        float val = row[(int64_t)s * D];
        if (val > max_val) max_val = val;
    }
    // Block reduction for max
    __shared__ float smax[256]; // assuming blockDim <= 256
    smax[threadIdx.x] = max_val;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            if (smax[threadIdx.x + offset] > smax[threadIdx.x]) {
                smax[threadIdx.x] = smax[threadIdx.x + offset];
            }
        }
        __syncthreads();
    }
    float block_max = smax[0];
    __syncthreads();

    // Step 2: compute sum of exp(x - max)
    float sum = 0.0f;
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        float val = row[(int64_t)s * D];
        sum += __expf(val - block_max);
    }
    // Block reduction for sum
    __shared__ float ssum[256];
    ssum[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            ssum[threadIdx.x] += ssum[threadIdx.x + offset];
        }
        __syncthreads();
    }
    float block_sum = ssum[0];
    __syncthreads();

    // Step 3: compute log_softmax = x - max - log(sum)
    float log_sum = logf(block_sum);
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
        float val = row[(int64_t)s * D];
        out_row[(int64_t)s * D] = val - block_max - log_sum;
    }
}

// Host launcher: assumes all pointers are CUDA device pointers, stream is valid.
// The binding layer must have already copied indices and weight to GPU, and allocated output tensors.
extern "C" void embedding_log_softmax_launcher(
    const int64_t* indices,
    const float* weight,
    float* embedding_out,
    float* log_softmax_out,
    int B, int S, int D, int V,
    cudaStream_t stream)
{
    if (B <= 0 || S <= 0 || D <= 0) return;

    // Launch embedding lookup
    int64_t total_elements = (int64_t)B * S * D;
    int block_size = 256;
    int64_t grid_size = (total_elements + block_size - 1) / block_size;
    // Cap grid size to avoid exceeding limits; use grid-stride loop
    int max_grid = 65535; // typical max for x dimension in many GPUs, but we can use 2D if needed; for simplicity use 1D with cap and grid-stride
    if (grid_size > max_grid) grid_size = max_grid;
    if (grid_size < 1) grid_size = 1;

    embedding_lookup_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        indices, weight, embedding_out, B, S, D);

    // Launch log_softmax: one block per (b,d) pair
    int total_pairs = B * D;
    int block_size_lg = 256; // 256 threads per block
    // Each block handles one (b,d) pair, grid size = total_pairs
    log_softmax_kernel<<<total_pairs, block_size_lg, 0, stream>>>(
        embedding_out, log_softmax_out, B, S, D);
}
