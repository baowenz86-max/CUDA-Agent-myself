// CUDA kernels for model_new custom ops: sum_dim1 and log_softmax_dim0
#include <cuda_runtime.h>
#include <float.h>

// -----------------------------------------------------------------------------
// sum_dim1: sum over dimension 1 of a 4D tensor [N, C, H, W]
// Input: x (float*) with strides sxN, sxC, sxH, sxW (elements, not bytes)
// Output: out (float*) shape [N, H, W] with strides soN, soH, soW
// Assumes out is allocated by caller with shape [N,H,W].
// -----------------------------------------------------------------------------
__global__ void sum_dim1_kernel(const float* __restrict__ x,
                                float* __restrict__ out,
                                int N, int C, int H, int W,
                                long long sxN, long long sxC, long long sxH, long long sxW,
                                long long soN, long long soH, long long soW,
                                long long total_out) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < total_out; i += stride) {
        int w = i % W;
        long long tmp = i / W;
        int h = tmp % H;
        int n = tmp / H;
        float acc = 0.0f;
        const float* x_base = x + (long long)n * sxN + (long long)h * sxH + (long long)w * sxW;
        for (int c = 0; c < C; ++c) {
            acc += x_base[c * sxC];
        }
        out[(long long)n * soN + (long long)h * soH + (long long)w * soW] = acc;
    }
}

extern "C" void sum_dim1_launcher(const float* x,
                                  float* out,
                                  int N, int C, int H, int W,
                                  long long sxN, long long sxC, long long sxH, long long sxW,
                                  long long soN, long long soH, long long soW,
                                  cudaStream_t stream) {
    long long total_out = (long long)N * H * W;
    if (total_out == 0 || C == 0) return;
    const int block = 256;
    int grid = (int)((total_out + block - 1) / block);
    if (grid > 65535) grid = 65535;
    sum_dim1_kernel<<<grid, block, 0, stream>>>(x, out, N, C, H, W,
                                                 sxN, sxC, sxH, sxW,
                                                 soN, soH, soW,
                                                 total_out);
}

// -----------------------------------------------------------------------------
// log_softmax_dim0: compute log_softmax over dimension 0 for a flattened [N, inner_size]
// Input: x (float*) shape [N, inner_size] with strides sxN, sxInner (contiguous or not)
// Output: out (float*) same shape, with strides soN, soInner
// Note: The contract says x is [N, ...] with dim=0, so we only need 2D view.
// The caller will provide N and inner_size = product of remaining dims, along with appropriate strides.
// -----------------------------------------------------------------------------

// Kernel to compute per-column max and sum of exp(x - max) using parallel reduction within block.
// Each block handles one column index (inner index j) across all N rows.
template <int BLOCK_SIZE>
__global__ void log_softmax_dim0_reduce_kernel(const float* __restrict__ x,
                                               float* __restrict__ partial_max,
                                               float* __restrict__ partial_sum,
                                               int N, long long sxN, long long sxInner,
                                               long long inner_size) {
    __shared__ float smem_max[BLOCK_SIZE];
    __shared__ float smem_sum[BLOCK_SIZE];
    long long j = blockIdx.x;
    if (j >= inner_size) return;
    int tid = threadIdx.x;
    // Each thread processes strided rows
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        float val = x[(long long)i * sxN + j * sxInner];
        if (val > local_max) local_max = val;
    }
    smem_max[tid] = local_max;
    __syncthreads();
    // Reduce max within block
    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smem_max[tid + s] > smem_max[tid]) smem_max[tid] = smem_max[tid + s];
        }
        __syncthreads();
    }
    float block_max = smem_max[0];
    __syncthreads();

    // Compute sum of exp(x - block_max)
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        float val = x[(long long)i * sxN + j * sxInner];
        local_sum += expf(val - block_max);
    }
    smem_sum[tid] = local_sum;
    __syncthreads();
    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            smem_sum[tid] += smem_sum[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_max[j] = block_max;
        partial_sum[j] = smem_sum[0];
    }
}

// Kernel to combine partial results and write log_softmax output.
// Ideally we would use a second kernel that computes global max and sum,
// but to keep a single .cu file and simplicity we use a single-block kernel
// to reduce partial arrays and another elementwise kernel.
// However, we can fuse the final computation into one kernel that first reduces partial_max and partial_sum
// in shared memory, then computes output for its assigned columns.
// To avoid extra kernel launches and synchronizations, we use a two-phase approach:
// Phase 1: log_softmax_dim0_reduce_kernel writes partial_max and partial_sum (size inner_size).
// Phase 2: log_softmax_dim0_final_kernel reads partial arrays, reduces them over columns to get global values,
//          and writes final output.
// But phase 2 must be launched after phase 1 completes; thus we need a host-side synchronization or separate stream.
// Since the launcher has only one stream, we can call cudaDeviceSynchronize between launches or use cooperative groups.
// For simplicity and correctness, we will launch the reduce kernel, then launch a small final kernel that reduces
// partial arrays using a single block, then launch the elementwise output kernel.
// To reduce synchronization cost, we can combine the final reduction and output into a single kernel where each
// block handles a chunk of columns, but each block would need to recompute the global max and sum redundantly.
// Given inner_size is 128*1024 = 131072, partial arrays are large (512KB each), and reducing them in one block
// may exceed shared memory. Better approach: use a two-pass algorithm with global reduction via atomics
// or a separate reduction kernel. But for simplicity, we can use a single kernel that computes the global max and sum
// per column using a grid-wide reduction with multiple blocks, then a final elementwise kernel.
// However, we can also compute log_softmax with a single kernel that uses shared memory per block for each column if
// inner_size is small enough? No, inner_size=131072, so one block per column is too many blocks (131072 blocks, each
// handles N=640 rows). That is actually okay: gridDim.x = inner_size, blockDim.x = some power of two >= N (640).
// Then each block can compute max and sum for its column entirely in shared memory and write output directly,
// eliminating the need for partial arrays and multiple kernels.
// Let's implement that optimized single-kernel version.

template <int BLOCK_SIZE>
__global__ void log_softmax_dim0_single_kernel(const float* __restrict__ x,
                                               float* __restrict__ out,
                                               int N, long long sxN, long long sxInner,
                                               long long soN, long long soInner,
                                               long long inner_size) {
    long long j = blockIdx.x;
    if (j >= inner_size) return;
    int tid = threadIdx.x;
    __shared__ float smem_max[BLOCK_SIZE];
    __shared__ float smem_sum[BLOCK_SIZE];

    // Compute max in this column (N elements)
    float local_max = -FLT_MAX;
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        float val = x[(long long)i * sxN + j * sxInner];
        if (val > local_max) local_max = val;
    }
    smem_max[tid] = local_max;
    __syncthreads();
    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            if (smem_max[tid + s] > smem_max[tid]) smem_max[tid] = smem_max[tid + s];
        }
        __syncthreads();
    }
    float col_max = smem_max[0];
    __syncthreads();  // ensure all threads read col_max after reduction

    // Compute sum of exp(x - col_max)
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        float val = x[(long long)i * sxN + j * sxInner];
        local_sum += expf(val - col_max);
    }
    smem_sum[tid] = local_sum;
    __syncthreads();
    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            smem_sum[tid] += smem_sum[tid + s];
        }
        __syncthreads();
    }
    float col_sum = smem_sum[0];
    float log_sum = logf(col_sum);
    __syncthreads();  // ensure all threads see col_sum and log_sum

    // Write output: out[i,j] = x[i,j] - col_max - log_sum
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        float val = x[(long long)i * sxN + j * sxInner];
        out[(long long)i * soN + j * soInner] = val - col_max - log_sum;
    }
}

extern "C" void log_softmax_dim0_launcher(const float* x,
                                          float* out,
                                          int N, long long inner_size,
                                          long long sxN, long long sxInner,
                                          long long soN, long long soInner,
                                          cudaStream_t stream) {
    if (N == 0 || inner_size == 0) return;
    // Choose block size: next power of two >= N, but not too large.
    int block = 256;
    while (block < N) block *= 2;
    if (block > 1024) block = 1024;  // cap to 1024 threads
    // If N > 1024, our loop handles strided access; block remains 1024.
    // grid = number of columns (inner_size), but cap to avoid too many blocks? We must launch inner_size blocks.
    // inner_size = 131072, which is acceptable (< 2^31).
    long long total_blocks = inner_size;
    if (total_blocks > 65535) {
        // gridDim.x max 2147483647, so 131072 is fine. No need grid-stride loop unless > 2^31-1.
    }
    // But if inner_size is huge, we could use grid-stride loop to limit grid size; but here 131072 blocks is fine.
    // We'll use a grid-stride loop for robustness, but it may reduce performance slightly.
    // For now, assume total_blocks <= max grid dimension.
    log_softmax_dim0_single_kernel<256><<<(unsigned int)total_blocks, 256, 0, stream>>>(
        x, out, N, sxN, sxInner, soN, soInner, inner_size);
}
