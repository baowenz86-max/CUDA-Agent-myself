#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

// Helper: block reduce max within a block using shared memory
template<typename T>
__device__ T blockReduceMax(T val, T* smem, int tid) {
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    return smem[0];
}

// Helper: block reduce sum within a block using shared memory
template<typename T>
__device__ T blockReduceSum(T val, T* smem, int tid) {
    smem[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    return smem[0];
}

// Kernel: log_softmax along dim=1
// Input: x shape [B, D, S] (here D = S = 1000), strides: sx_b, sx_d, sx_s
// Output: y same shape, strides sy_b, sy_d, sy_s
// Each block handles one (batch, class) row: b, j. Threads iterate over dim1 (i).
// We compute max over the row first, then sum of exp(x - max), then write y = x - max - log(sum).
__global__ void log_softmax_kernel(const float* __restrict__ x,
                                   float* __restrict__ y,
                                   int D,
                                   int S,
                                   long long sx_b,
                                   long long sx_d,
                                   long long sx_s,
                                   long long sy_b,
                                   long long sy_d,
                                   long long sy_s) {
    // Block index: b = blockIdx.y, j = blockIdx.x? Or we can use one-dimensional grid mapping to total rows.
    // We have total rows = B * S. Let's map blockIdx.x to row index.
    int total_rows = blockDim.x * gridDim.x; // not used; use blockIdx.x directly
    int row = blockIdx.x; // one block per row
    if (row >= gridDim.x) return;

    // Decode b and j from row: row = b * S + j
    int b = row / S;
    int j = row - b * S;

    // Shared memory for reduction
    extern __shared__ float smem[];
    float* max_smem = smem;                 // blockDim.x floats
    float* sum_smem = smem + blockDim.x;    // blockDim.x floats

    int tid = threadIdx.x;
    int nt = blockDim.x;

    // Phase 1: compute max over i=0..D-1
    float max_val = -FLT_MAX;
    for (int i = tid; i < D; i += nt) {
        long long idx = (long long)b * sx_b + (long long)i * sx_d + (long long)j * sx_s;
        float val = x[idx];
        max_val = fmaxf(max_val, val);
    }
    max_val = blockReduceMax(max_val, max_smem, tid);

    // Phase 2: compute sum of exp(x - max) over i
    float sum_val = 0.0f;
    for (int i = tid; i < D; i += nt) {
        long long idx = (long long)b * sx_b + (long long)i * sx_d + (long long)j * sx_s;
        float val = __expf(x[idx] - max_val);
        sum_val += val;
    }
    sum_val = blockReduceSum(sum_val, sum_smem, tid);
    float log_sum = logf(sum_val);

    // Phase 3: write output y = x - max - log_sum
    for (int i = tid; i < D; i += nt) {
        long long x_idx = (long long)b * sx_b + (long long)i * sx_d + (long long)j * sx_s;
        long long y_idx = (long long)b * sy_b + (long long)i * sy_d + (long long)j * sy_s;
        y[y_idx] = x[x_idx] - max_val - log_sum;
    }
}

// Kernel: argmax along dim=1
// Input: x shape [B, D, S], strides sx_b, sx_d, sx_s
// Output: out shape [B, S], dtype int64_t
// Each block handles one (b, j) row.
__global__ void argmax_kernel(const float* __restrict__ x,
                              int64_t* __restrict__ out,
                              int B,
                              int D,
                              int S,
                              long long sx_b,
                              long long sx_d,
                              long long sx_s,
                              long long so_b,
                              long long so_s) {
    int row = blockIdx.x;
    if (row >= B * S) return;
    int b = row / S;
    int j = row - b * S;

    int tid = threadIdx.x;
    int nt = blockDim.x;

    // Compute max and argmax using block reduction
    // We'll use shared memory for two arrays: max and index
    extern __shared__ float smem[];
    float* max_smem = smem;
    int* idx_smem = (int*)(smem + blockDim.x); // store int indices

    float local_max = -FLT_MAX;
    int local_idx = -1;
    for (int i = tid; i < D; i += nt) {
        long long x_idx = (long long)b * sx_b + (long long)i * sx_d + (long long)j * sx_s;
        float val = x[x_idx];
        if (val > local_max) {
            local_max = val;
            local_idx = i;
        }
    }

    max_smem[tid] = local_max;
    idx_smem[tid] = local_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (max_smem[tid + s] > max_smem[tid]) {
                max_smem[tid] = max_smem[tid + s];
                idx_smem[tid] = idx_smem[tid + s];
            } else if (max_smem[tid + s] == max_smem[tid]) {
                // Tie-breaking: PyTorch argmax returns the first occurrence.
                // Since we process lower indices first, we should keep the lower index.
                // Our local_idx for each thread is the first max within that thread's subset.
                // In reduction, if equal, keep the one with smaller index.
                if (idx_smem[tid + s] < idx_smem[tid]) {
                    idx_smem[tid] = idx_smem[tid + s];
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        long long out_idx = (long long)b * so_b + (long long)j * so_s;
        out[out_idx] = (int64_t)idx_smem[0];
    }
}

// Host launcher for log_softmax
// x: pointer to input tensor, y: pointer to output tensor
// B, D, S: dimensions (batch, rows, cols)
// strides: input strides in elements, output strides in elements
// stream: CUDA stream
extern "C" void log_softmax_forward_launcher(const float* x,
                                             float* y,
                                             int B,
                                             int D,
                                             int S,
                                             long long sx_b,
                                             long long sx_d,
                                             long long sx_s,
                                             long long sy_b,
                                             long long sy_d,
                                             long long sy_s,
                                             cudaStream_t stream) {
    int rows = B * S; // total rows to process
    int block_size = 256;
    int grid_size = rows; // one block per row; rows = 200,000 fits within grid limits for CUDA (max 2^31-1)
    // Shared memory: two arrays of block_size floats for reductions, plus for argmax we need extra for indices.
    // For log_softmax: 2 * block_size * sizeof(float)
    size_t shared_mem = 2 * block_size * sizeof(float);
    log_softmax_kernel<<<grid_size, block_size, shared_mem, stream>>>(
        x, y, D, S, sx_b, sx_d, sx_s, sy_b, sy_d, sy_s);
}

// Host launcher for argmax
// x: input pointer, out: output pointer (int64_t)
// B, D, S, strides
// stream: CUDA stream
extern "C" void argmax_forward_launcher(const float* x,
                                        int64_t* out,
                                        int B,
                                        int D,
                                        int S,
                                        long long sx_b,
                                        long long sx_d,
                                        long long sx_s,
                                        long long so_b,
                                        long long so_s,
                                        cudaStream_t stream) {
    int rows = B * S;
    int block_size = 256;
    int grid_size = rows;
    // Shared memory: max (float) + idx (int) blocks, each block_size
    size_t shared_mem = block_size * sizeof(float) + block_size * sizeof(int);
    argmax_kernel<<<grid_size, block_size, shared_mem, stream>>>(
        x, out, B, D, S, sx_b, sx_d, sx_s, so_b, so_s);
}
