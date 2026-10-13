#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

extern "C" {

// -----------------------------------------------------------------------------
// Utility device functions
// -----------------------------------------------------------------------------
__device__ __forceinline__ float softmax_warp_reduce(float val) {
    // Warp-level reduction for max and sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    // Reduce sum across a warp and then across warps using shared memory
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();
    // Block reduce (first warp)
    if (threadIdx.x < 32) {
        val = (threadIdx.x < (blockDim.x >> 5)) ? warp_sums[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0) warp_sums[0] = val;
    }
    __syncthreads();
    return warp_sums[0];
}

// -----------------------------------------------------------------------------
// Kernel: in_projection_forward
// Computes q = query @ W_q^T + b_q, k = key @ W_k^T + b_k, v = value @ W_v^T + b_v
// Inputs: query, key, value: [batch, seq_len, embed_dim]
// in_proj_weight: [3*embed_dim, embed_dim]  (stored in row-major)
// in_proj_bias: [3*embed_dim] or nullptr if no bias
// Outputs: q, k, v: [batch, seq_len, embed_dim]
// We use a grid-stride loop over batch*seq_len*embed_dim output elements,
// computing dot products for each output element.
// For performance, we'll use one block per row (batch*seq_len) and threads over embed_dim,
// with shared memory caching the weight row.
// However, weight rows are only embed_dim long, so we can have each thread compute one output element,
// but we need to compute three outputs per input row, each using different weight slices.
// We'll launch threads in x dimension for embed_dim, and blockIdx.x for batch*seq_len, with three separate kernels or a single kernel that computes all three outputs.
// Simpler: one kernel that computes all three outputs by having each thread handle one output element.
// Total output elements: 3 * batch * seq_len * embed_dim.
// We'll use grid-stride loop over the flattened index.
//
__global__ void in_projection_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out_q,
    float* __restrict__ out_k,
    float* __restrict__ out_v,
    int batch,
    int seq_len,
    int embed_dim,
    int total_batch_seq) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int num_outputs_per_input = 3 * embed_dim;
    int total_outputs = total_batch_seq * num_outputs_per_input;
    bool has_bias = (bias != nullptr);

    for (int i = idx; i < total_outputs; i += stride) {
        int out_idx_in_seq = i % num_outputs_per_input; // 0..3*embed_dim-1
        int row = i / num_outputs_per_input; // 0..total_batch_seq-1
        int which = out_idx_in_seq / embed_dim; // 0: q, 1: k, 2: v
        int col = out_idx_in_seq % embed_dim;   // output feature index

        // input pointer: which == 0 -> query, 1 -> key, 2 -> value
        const float* input = (which == 0) ? query : (which == 1) ? key : value;
        // weight offset: row = which*embed_dim + col
        const float* w_row = weight + (which * embed_dim + col) * embed_dim;

        float sum = 0.0f;
        for (int e = 0; e < embed_dim; ++e) {
            sum += input[row * embed_dim + e] * w_row[e];
        }
        if (has_bias) {
            sum += bias[which * embed_dim + col];
        }

        // write to appropriate output
        if (which == 0) {
            out_q[row * embed_dim + col] = sum;
        } else if (which == 1) {
            out_k[row * embed_dim + col] = sum;
        } else {
            out_v[row * embed_dim + col] = sum;
        }
    }
}

void in_projection_forward_launcher(
    const float* query,
    const float* key,
    const float* value,
    const float* in_proj_weight,
    const float* in_proj_bias,
    float* out_q,
    float* out_k,
    float* out_v,
    int batch,
    int seq_len,
    int embed_dim,
    cudaStream_t stream) {

    int total_batch_seq = batch * seq_len;
    int num_outputs_per_input = 3 * embed_dim;
    int total_outputs = total_batch_seq * num_outputs_per_input;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid limit; use grid-stride loop
    in_projection_kernel<<<blocks, threads, 0, stream>>>(
        query, key, value, in_proj_weight, in_proj_bias,
        out_q, out_k, out_v,
        batch, seq_len, embed_dim, total_batch_seq);
}

// -----------------------------------------------------------------------------
// Kernel: attention_weights_forward
// Inputs: q, k: [batch, num_heads, seq_len, head_dim] (contiguous)
// num_heads: int, head_dim: int, seq_len known from shape or passed? In launcher we pass all dims.
// Output: attn_weights_avg: [batch, seq_len, seq_len]  (averaged over heads)
// Algorithm: for each batch b and head h, compute scores = q_bh @ k_bh^T / sqrt(head_dim), softmax over last dim,
// then accumulate into averaging buffer. We'll do one thread block per (b, seq_len, seq_len?) Actually we need output [batch, seq_len, seq_len];
// We can have each thread compute one output element (i,j) by iterating over heads and summing their softmax outputs, then divide by num_heads.
// For each head, we need to compute q_i dot k_j, scaled, then softmax over j for fixed i.
// Softmax requires max and sum across j. We'll compute per head per row i: first compute scores for all j into shared memory or registers, then softmax, then sum across heads.
// For efficiency, each block handles one head and one batch, and one row i (or multiple rows).
// To keep it simpler, we'll use one block per (batch, head, i) and have threads over j to compute scores, softmax, and then atomicAdd to output.
// Since num_heads=8, batch=16, seq_len=1024, total blocks = 16*8*1024 = 131072, which exceeds grid limit in one dimension, but we can use 2D grid.
// We'll use grid: (batch * num_heads, seq_len) and block size = seq_len (or 1024, or multiple waves).
// Each block computes one row of attention weights for one head, and atomically adds to output after scaling by 1/num_heads.
// However, atomicAdd on float may have non-deterministic order, but the sum across heads is commutative, so fine.
// Alternatively, compute all heads for a given (batch, i) in one block to avoid atomics, but that would require more shared memory.
// We'll implement atomic version for simplicity and correctness.

__global__ void attention_weights_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ out_avg,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale) {

    // blockIdx.x: batch*num_heads + head  (flattened)
    int linear_bh = blockIdx.x;
    int batch = linear_bh / num_heads;
    int head = linear_bh % num_heads;
    int i = blockIdx.y; // query position (seq index)
    int tid = threadIdx.x;
    int threads = blockDim.x;

    // Base pointers for this batch and head
    const float* q_bh = q + ((batch * num_heads + head) * seq_len + i) * head_dim;
    const float* k_bh = k + (batch * num_heads + head) * seq_len * head_dim;

    // Shared memory for scores (seq_len floats)
    extern __shared__ float scores[]; // size seq_len

    // Compute scores for this i against all j
    for (int j = tid; j < seq_len; j += threads) {
        float dot = 0.0f;
        const float* k_row = k_bh + j * head_dim;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_bh[d] * k_row[d];
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    // Compute max for softmax
    float max_val = -1e30f;
    for (int j = tid; j < seq_len; j += threads) {
        max_val = fmaxf(max_val, scores[j]);
    }
    // Warp reduce max within block
    // We'll use a simple block reduction using shared memory
    __shared__ float block_max[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    // Reduce within warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    if (lane == 0) block_max[warp_id] = max_val;
    __syncthreads();
    if (tid < 32) {
        max_val = (tid < (threads >> 5)) ? block_max[tid] : -1e30f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
        }
        if (tid == 0) block_max[0] = max_val;
    }
    __syncthreads();
    max_val = block_max[0];

    // Compute exp and sum
    float sum = 0.0f;
    for (int j = tid; j < seq_len; j += threads) {
        float e = __expf(scores[j] - max_val);
        scores[j] = e;
        sum += e;
    }
    // Reduce sum
    __shared__ float block_sum[32];
    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) block_sum[warp_id] = sum;
    __syncthreads();
    if (tid < 32) {
        sum = (tid < (threads >> 5)) ? block_sum[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) block_sum[0] = sum;
    }
    __syncthreads();
    sum = block_sum[0];

    // Normalize and add to output (average over heads)
    float inv_sum = 1.0f / sum;
    float inv_heads = 1.0f / (float)num_heads;
    for (int j = tid; j < seq_len; j += threads) {
        float attn = scores[j] * inv_sum * inv_heads;
        // atomicAdd to out_avg[batch, i, j]
        atomicAdd(&out_avg[(batch * seq_len + i) * seq_len + j], attn);
    }
}

void attention_weights_forward_launcher(
    const float* q,
    const float* k,
    float* out_avg,
    int batch,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream) {

    float scale = 1.0f / sqrtf((float)head_dim);
    int blocks_x = batch * num_heads;
    int blocks_y = seq_len;
    dim3 grid(blocks_x, blocks_y);
    int threads = 1024; // we'll use grid-stride loop over j
    int shared_mem_size = seq_len * sizeof(float); // scores array
    // Ensure shared memory fits
    if (shared_mem_size > 48*1024) {
        // fallback: use smaller block and maybe less shared? but seq_len=1024 -> 4KB, fine.
        shared_mem_size = seq_len * sizeof(float); // 4KB, ok
    }
    attention_weights_kernel<<<grid, threads, shared_mem_size, stream>>>(
        q, k, out_avg, num_heads, seq_len, head_dim, scale);
}

// -----------------------------------------------------------------------------
// Kernel: output_projection_forward
// Computes output = attn_output @ out_proj_weight^T + out_proj_bias
// Input: attn_output [batch, seq_len, embed_dim]
// out_proj_weight [embed_dim, embed_dim]
// out_proj_bias [embed_dim] or nullptr
// Output: [batch, seq_len, embed_dim]
// Similar to in_projection but single output.
__global__ void output_projection_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int total_rows,
    int embed_dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total_outputs = total_rows * embed_dim;
    bool has_bias = (bias != nullptr);

    for (int i = idx; i < total_outputs; i += stride) {
        int row = i / embed_dim;
        int col = i % embed_dim;
        const float* w_row = weight + col * embed_dim; // because weight is [embed_dim, embed_dim], row-major: weight[col*embed_dim + d]
        float sum = 0.0f;
        const float* in_row = input + row * embed_dim;
        for (int d = 0; d < embed_dim; ++d) {
            sum += in_row[d] * w_row[d];
        }
        if (has_bias) sum += bias[col];
        output[i] = sum;
    }
}

void output_projection_forward_launcher(
    const float* input,
    const float* out_proj_weight,
    const float* out_proj_bias,
    float* output,
    int batch,
    int seq_len,
    int embed_dim,
    cudaStream_t stream) {

    int total_rows = batch * seq_len;
    int total_outputs = total_rows * embed_dim;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    output_projection_kernel<<<blocks, threads, 0, stream>>>(
        input, out_proj_weight, out_proj_bias, output, total_rows, embed_dim);
}

// -----------------------------------------------------------------------------
// Kernel: smooth_l1_loss_forward
// Computes smooth L1 loss with reduction='mean' over all elements.
// Input: input, target: [batch, seq_len, 2*embed_dim] (same shape)
// beta: float
// Output: scalar tensor (we return a single float value in a preallocated 1-element tensor)
// We'll use a grid-stride loop and atomicAdd to a global accumulator, then divide by total elements.
// Since we need mean, we can either accumulate sum and then divide, or use atomicAdd on the output directly.
// We'll allocate a zero-initialized output float, and then after kernel, divide by total_count.
// The binding will handle the division.

__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float beta,
    float* __restrict__ loss_out,
    int total_count) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // We'll use a shared memory per-block reduction to avoid many atomics.
    __shared__ float block_sum[256]; // assuming block size <=256? We'll use dynamic block size, but for simplicity we'll use fixed 256 threads.
    // We'll hardcode threads as 256 in launcher, so shared size is fine.
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    for (int i = idx; i < total_count; i += stride) {
        float diff = input[i] - target[i];
        float abs_diff = fabsf(diff);
        float loss;
        if (abs_diff < beta) {
            loss = 0.5f * diff * diff / beta;
        } else {
            loss = abs_diff - 0.5f * beta;
        }
        local_sum += loss;
    }

    // Reduce within block
    block_sum[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            block_sum[tid] += block_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss_out, block_sum[0]);
    }
}

void smooth_l1_loss_forward_launcher(
    const float* input,
    const float* target,
    float beta,
    float* loss_out,
    int total_count,
    cudaStream_t stream) {

    // Initialize loss_out to zero before kernel? We'll let binding do cudaMemset or assume zero.
    // But we can also set it to zero here by a single thread in a separate kernel, but better to have binding do it.
    int threads = 256;
    int blocks = (total_count + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    smooth_l1_loss_kernel<<<blocks, threads, 0, stream>>>(
        input, target, beta, loss_out, total_count);
}

// -----------------------------------------------------------------------------
// Launcher declarations with C linkage (for binding)
// These must match the expected signatures in custom_ops_contract.json.
// We'll define functions that match the operator names.

// 1. in_projection_forward: inputs query, key, value, in_proj_weight, in_proj_bias (nullable)
//    Outputs: three tensors q, k, v.
//    Launcher needs to receive output pointers preallocated.
//    We'll define a struct-like return? But binding will allocate outputs and pass pointers.
//    So launcher signature: (query, key, value, weight, bias, out_q, out_k, out_v, batch, seq_len, embed_dim, stream)

void in_projection_forward(
    const float* query,
    const float* key,
    const float* value,
    const float* in_proj_weight,
    const float* in_proj_bias,
    float* out_q,
    float* out_k,
    float* out_v,
    int batch,
    int seq_len,
    int embed_dim,
    cudaStream_t stream) {
    in_projection_forward_launcher(query, key, value, in_proj_weight, in_proj_bias, out_q, out_k, out_v, batch, seq_len, embed_dim, stream);
}

// 2. attention_weights_forward: inputs q, k, num_heads, head_dim
//    Output: attn_weights_avg [batch, seq_len, seq_len]
void attention_weights_forward(
    const float* q,
    const float* k,
    int num_heads,
    int head_dim,
    float* out_avg,
    int batch,
    int seq_len,
    cudaStream_t stream) {
    attention_weights_forward_launcher(q, k, out_avg, batch, num_heads, seq_len, head_dim, stream);
}

// 3. output_projection_forward: inputs attn_output, out_proj_weight, out_proj_bias (nullable)
//    Output: output [batch, seq_len, embed_dim]
void output_projection_forward(
    const float* input,
    const float* out_proj_weight,
    const float* out_proj_bias,
    float* output,
    int batch,
    int seq_len,
    int embed_dim,
    cudaStream_t stream) {
    output_projection_forward_launcher(input, out_proj_weight, out_proj_bias, output, batch, seq_len, embed_dim, stream);
}

// 4. smooth_l1_loss_forward: inputs input, target, beta, output scalar pointer
//    Output: scalar tensor (float)
void smooth_l1_loss_forward(
    const float* input,
    const float* target,
    float beta,
    float* loss_out,
    int total_count,
    cudaStream_t stream) {
    // Reset loss_out to 0 before launching
    cudaMemsetAsync(loss_out, 0, sizeof(float), stream);
    smooth_l1_loss_forward_launcher(input, target, beta, loss_out, total_count, stream);
}

} // extern "C"
