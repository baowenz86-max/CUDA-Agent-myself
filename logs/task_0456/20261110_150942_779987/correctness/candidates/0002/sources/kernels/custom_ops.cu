#include <cuda_runtime.h>
#include <math.h>

// Batchnorm inference kernel
// Each thread computes one output element (NCHW contiguous)
__global__ void batchnorm_inference_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    const float* __restrict__ weight, // may be null
    const float* __restrict__ bias,   // may be null
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N,
    int C,
    int HW
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < (int64_t)N * C * HW; i += stride) {
        int n = i / (C * HW);
        int c = (i / HW) % C;
        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float gamma = (weight != nullptr) ? weight[c] : 1.0f;
        float beta = (bias != nullptr) ? bias[c] : 0.0f;
        float val = (x[i] - mean) * inv_std * gamma + beta;
        out[i] = val;
    }
}

// Triplet margin loss sum kernel
// Each block processes a subset of batch items and accumulates partial sum
// We assume batch size N = 16, but code is general.
// Each thread computes distance for one item? Better: one thread per item, reduce within block.
// Since feature dimension D = C*HW = 768*64*64 = 3,145,728, we cannot have one thread per item efficiently.
// We'll use one block per item, each thread sums over a chunk of D, then reduce across threads.
// To reduce code complexity, use one block per item, with 256 threads, grid-stride over D, and shared memory reduction.

template<int BLOCK_DIM>
__global__ void triplet_margin_loss_sum_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ partial_sums, // length = N (one per item)
    float margin,
    int N,
    int D
) {
    int item = blockIdx.x;
    if (item >= N) return;

    const float* a = anchor + (int64_t)item * D;
    const float* p = positive + (int64_t)item * D;
    const float* n = negative + (int64_t)item * D;

    float sum_pos = 0.0f;
    float sum_neg = 0.0f;

    // Each thread accumulates over a strided chunk
    for (int idx = threadIdx.x; idx < D; idx += BLOCK_DIM) {
        float diff_pos = a[idx] - p[idx];
        float diff_neg = a[idx] - n[idx];
        sum_pos += diff_pos * diff_pos;
        sum_neg += diff_neg * diff_neg;
    }

    // Reduce sums within block using shared memory
    __shared__ float s_pos[BLOCK_DIM];
    __shared__ float s_neg[BLOCK_DIM];
    s_pos[threadIdx.x] = sum_pos;
    s_neg[threadIdx.x] = sum_neg;
    __syncthreads();

    // Tree reduction
    for (int s = BLOCK_DIM / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_pos[threadIdx.x] += s_pos[threadIdx.x + s];
            s_neg[threadIdx.x] += s_neg[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float dist_pos = sqrtf(s_pos[0]);
        float dist_neg = sqrtf(s_neg[0]);
        float loss_i = fmaxf(dist_pos - dist_neg + margin, 0.0f);
        partial_sums[item] = loss_i;
    }
}

// Final reduction kernel: sum partial_sums into output scalar
__global__ void reduce_sum_kernel(
    const float* __restrict__ partial_sums,
    float* __restrict__ out,
    int N
) {
    // Single block, N items
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    if (tid < N) {
        sum = partial_sums[tid];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[0] = sdata[0];
    }
}

// Forward declaration of atomic kernel to be used in launcher
template<int BLOCK_DIM>
__global__ void triplet_margin_loss_sum_atomic_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ out,
    float margin,
    int N,
    int D
);

// Launcher functions
extern "C" void batchnorm_inference_launcher(
    const float* x,
    float* out,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N,
    int C,
    int HW,
    cudaStream_t stream
) {
    int64_t total = (int64_t)N * C * HW;
    int block_size = 256;
    int grid_size = (int)min((total + block_size - 1) / block_size, (int64_t)65535);
    if (grid_size < 1) grid_size = 1;
    batchnorm_inference_kernel<<<grid_size, block_size, 0, stream>>>(
        x, out, weight, bias, running_mean, running_var, eps, N, C, HW);
}

extern "C" void triplet_margin_loss_sum_launcher(
    const float* anchor,
    const float* positive,
    const float* negative,
    float* out,
    float margin,
    int N,
    int D,
    cudaStream_t stream
) {
    // Allocate partial sums on device? We need a workspace.
    // Better: use a single kernel with atomicAdd, but order may be non-deterministic but acceptable for sum of floats.
    // However, requirement says use stable scan? Actually masked_select specific, not this.
    // We can use an internal temporary buffer allocated in launcher with cudaMalloc, but that requires cudaFree.
    // Simpler: use atomicAdd in a single kernel where each block computes one item and atomically adds to output.
    // Since reduction='sum', atomicAdd is fine for floating point? It is deterministic up to floating point non-associativity, acceptable.
    // Let's implement a kernel where each block does the per-item distance computation and then atomically adds to out[0].
    // First zero out[0] using cudaMemsetAsync in launcher.
    // We'll write a kernel that does per-item block and atomicAdd.
    
    // We'll define a kernel: one block per item, threads sum over D, then one thread atomically adds loss_i to out.
    // Instead of partial_sums and separate reduction.
    
    constexpr int BLOCK_DIM = 256;
    // Ensure D is divisible or handle remainder
    // We'll use a loop as above.
    
    // Launch zeroing kernel? Use cudaMemsetAsync before main kernel.
    cudaMemsetAsync(out, 0, sizeof(float), stream);
    
    int num_items = N;
    triplet_margin_loss_sum_atomic_kernel<BLOCK_DIM><<<num_items, BLOCK_DIM, 0, stream>>>(
        anchor, positive, negative, out, margin, N, D);
}

// Kernel with atomic add directly to output scalar
// Shared memory reduction then one thread does atomicAdd
template<int BLOCK_DIM>
__global__ void triplet_margin_loss_sum_atomic_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ out,
    float margin,
    int N,
    int D
) {
    int item = blockIdx.x;
    if (item >= N) return;

    const float* a = anchor + (int64_t)item * D;
    const float* p = positive + (int64_t)item * D;
    const float* n = negative + (int64_t)item * D;

    float sum_pos = 0.0f;
    float sum_neg = 0.0f;

    for (int idx = threadIdx.x; idx < D; idx += BLOCK_DIM) {
        float diff_pos = a[idx] - p[idx];
        float diff_neg = a[idx] - n[idx];
        sum_pos += diff_pos * diff_pos;
        sum_neg += diff_neg * diff_neg;
    }

    __shared__ float s_pos[BLOCK_DIM];
    __shared__ float s_neg[BLOCK_DIM];
    s_pos[threadIdx.x] = sum_pos;
    s_neg[threadIdx.x] = sum_neg;
    __syncthreads();

    for (int s = BLOCK_DIM / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_pos[threadIdx.x] += s_pos[threadIdx.x + s];
            s_neg[threadIdx.x] += s_neg[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float dist_pos = sqrtf(s_pos[0]);
        float dist_neg = sqrtf(s_neg[0]);
        float loss_i = fmaxf(dist_pos - dist_neg + margin, 0.0f);
        atomicAdd(out, loss_i);
    }
}
