// kernels/custom_ops.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <cstdint>

#define CHECK_CUDA(call) { cudaError_t err = call; if (err != cudaSuccess) { printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); return; } }
#define CHECK_CUBLAS(call) { cublasStatus_t st = call; if (st != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS error at %s:%d\n", __FILE__, __LINE__); return; } }

// ---------------------------------------------------------------------
// Helper: scaled dot-product attention kernel
// Input tensors are assumed to be in batch_first format: [B, S, E]
// We compute attention per batch and head.
// For each batch b and head h:
//   q = Q[b, :, h, :]  (S x D)
//   k = K[b, :, h, :]  (S x D)
//   v = V[b, :, h, :]  (S x D)
//   scores = q @ k^T / sqrt(D)
//   probs = softmax(scores, dim=-1)
//   out = probs @ v
//   attn_weights[b,h,:,:] = probs
// Output attn_output: [B, S, E] (contiguous)
// Output attn_output_weights: [B, S, S] (contiguous) averaged over heads
// ---------------------------------------------------------------------

__global__ void attention_forward_kernel(
    const float* __restrict__ q,       // [B, S, E]
    const float* __restrict__ k,       // [B, S, E]
    const float* __restrict__ v,       // [B, S, E]
    float* __restrict__ attn_out,      // [B, S, E]
    float* __restrict__ attn_weights,  // [B, S, S] (sum over heads, will average later)
    int B, int S, int E, int H,        // batch, seq, embed, heads
    int D,                             // head_dim = E/H
    float scale                        // 1/sqrt(D)
) {
    // Each thread block processes one (batch, head) pair and one query position? 
    // Better: each block processes one (batch, head) and tiles over S.
    // For simplicity and correctness, use one thread per output element in attn_weights and attn_out.
    // But attention requires softmax over row, so we need per-row reduction.
    // We'll use a cooperative approach: each warp handles one row (query) for a given head and batch.
    // However, to keep it simple and correct, we can have one block per (batch, head, query), with blockDim.x = S (or multiples).
    // Since S=1024, that's feasible with 64 blocks of 1024 threads? Not exactly, max threads per block is 1024.
    // We'll use blockDim.x = 256 and loop over key positions.
    // Each thread computes partial dot products for a subset of key positions.
    // Thread reduction for max and sum.
    // Shared memory: for scores of row.
    extern __shared__ float shared_mem[]; // size: S * sizeof(float) for scores, max, sum
    float* scores = shared_mem;           // S floats
    // No need for separate max/sum, we can use shared for reductions.
    
    int batch = blockIdx.z;          // B
    int head = blockIdx.y;           // H
    int query_idx = blockIdx.x;      // query index (0..S-1)
    
    if (batch >= B || head >= H || query_idx >= S) return;
    
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    
    // Global pointers for this batch and head
    // q[b, s, h*D + d] because we have packed heads after projection: q = [B, S, H*D]
    // But our q is [B, S, E] with E = H*D, and we need to select head h.
    const float* q_ptr = q + (batch * S + query_idx) * E + head * D;
    const float* k_base = k + batch * S * E;
    const float* v_base = v + batch * S * E;
    
    // Step 1: compute scores for all key positions, store in shared scores[]
    // scores[j] = dot(q, k[j]) * scale
    // Use grid-stride loop over key positions
    for (int j = tid; j < S; j += nthreads) {
        const float* k_ptr = k_base + j * E + head * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += q_ptr[d] * k_ptr[d];
        }
        scores[j] = dot * scale;
    }
    __syncthreads();
    
    // Step 2: find max score for numerical stability
    float max_score = -INFINITY;
    for (int j = tid; j < S; j += nthreads) {
        max_score = fmaxf(max_score, scores[j]);
    }
    // Reduce max within block using shared memory or warp shuffle.
    // For simplicity, use shared memory reduction.
    // We'll use a simple warp shuffle reduction within a warp, then across warps with shared.
    // For simplicity, since blockDim.x <= 1024, we can do a warp shuffle.
    // But we need to handle blockDim not multiple of 32. We'll use a simple shared reduction with two-phase.
    // Reuse scores for reduction? We'll allocate additional shared for max and sum.
    // Let's allocate total shared as S * sizeof(float) + 2 * sizeof(float) for reductions.
    float* reduce_mem = scores + S; // use this for max and sum
    float* max_shared = reduce_mem;
    float* sum_shared = reduce_mem + 1;
    
    // Initialize
    if (tid == 0) {
        *max_shared = -INFINITY;
        *sum_shared = 0.0f;
    }
    __syncthreads();
    
    // Each thread has partial max, do atomic max? Use shared memory reduction with atomics is ok for max.
    // But simpler: use a second pass with manual reduction.
    // Let's do warp reduction.
    float local_max = max_score;
    // Warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, local_max, offset);
        local_max = fmaxf(local_max, other);
    }
    int warp_id = tid / 32;
    int lane = tid % 32;
    if (lane == 0) {
        // write warp max to shared
        max_shared[warp_id] = local_max;
    }
    __syncthreads();
    int num_warps = (nthreads + 31) / 32;
    if (tid == 0) {
        float block_max = -INFINITY;
        for (int i = 0; i < num_warps; ++i) {
            block_max = fmaxf(block_max, max_shared[i]);
        }
        *max_shared = block_max;
    }
    __syncthreads();
    float global_max = *max_shared;
    
    // Step 3: compute exp and sum
    float local_sum = 0.0f;
    for (int j = tid; j < S; j += nthreads) {
        float exp_val = expf(scores[j] - global_max);
        scores[j] = exp_val; // reuse scores to store exp values
        local_sum += exp_val;
    }
    // Reduce sum
    // Warp reduce sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) {
        sum_shared[warp_id] = local_sum;
    }
    __syncthreads();
    if (tid == 0) {
        float block_sum = 0.0f;
        for (int i = 0; i < num_warps; ++i) {
            block_sum += sum_shared[i];
        }
        *sum_shared = block_sum;
    }
    __syncthreads();
    float global_sum = *sum_shared;
    float inv_sum = 1.0f / global_sum;
    
    // Step 4: compute attention output for this query
    // out[d] = sum_j probs[j] * v[j, d]
    // We need to compute for all d in 0..D-1.
    // Each thread handles a subset of d.
    float* out_ptr = attn_out + (batch * S + query_idx) * E + head * D;
    
    // Compute weighted sum for each d assigned to this thread
    for (int d = tid; d < D; d += nthreads) {
        float acc = 0.0f;
        for (int j = 0; j < S; ++j) {
            float prob = scores[j] * inv_sum;
            const float* v_ptr = v_base + j * E + head * D + d;
            acc += prob * (*v_ptr);
        }
        out_ptr[d] = acc;
    }
    
    // Step 5: store attention weights (probs) into attn_weights, accumulate for averaging over heads.
    // We need to add probs for this head to attn_weights[b, query_idx, j].
    // To avoid race between heads, we write to a temporary per-head location or use atomicAdd to sum.
    // Since attn_weights is [B,S,S], we can use atomicAdd on float.
    // However, atomicAdd may be slower but acceptable.
    // We'll atomicAdd probs to attn_weights after dividing by H? But we can divide later on host or in separate kernel.
    // Better: accumulate sum, then define average = sum / H. We can do the division in the same kernel if we know H and only one block per head? But multiple heads write to same location, so we can't just write. We'll do atomicAdd sum, then have a separate kernel to divide by H, or incorporate division in atomicAdd by adding prob/H. But atomicAdd with float division is fine.
    // We'll add prob * (1.0f / H) to attn_weights, so final result is average.
    for (int j = tid; j < S; j += nthreads) {
        float prob = scores[j] * inv_sum;
        atomicAdd(&attn_weights[(batch * S + query_idx) * S + j], prob * (1.0f / H));
    }
}

// ---------------------------------------------------------------------
// Launcher for multihead attention forward
// Inputs: query, key, value: [B, S, E] float32 contiguous
// in_proj_weight: [3*E, E] (W_q concatenated W_k W_v)
// in_proj_bias: [3*E]
// out_proj_weight: [E, E]
// out_proj_bias: [E]
// num_heads: int
// batch_first: bool (assumed true, but handle if false)
// Outputs: attn_output [B, S, E], attn_output_weights [B, S, S]
// Both allocated by caller (binding) with torch::empty_like etc.
extern "C" void multihead_attention_forward_launcher(
    const float* query, const float* key, const float* value,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    float* attn_output, float* attn_output_weights,
    int B, int S, int E, int num_heads, int batch_first,
    cudaStream_t stream)
{
    int H = num_heads;
    int D = E / H;
    float scale = 1.0f / sqrtf((float)D);
    
    // We need temporary buffers for Q, K, V after in_projection: each [B, S, E]
    // Also output of attention before out_projection: [B, S, E]
    // We'll allocate device memory using cudaMalloc, then free.
    // For performance, could use cudaMallocAsync but we'll use cudaMalloc.
    size_t proj_size = (size_t)B * S * E * sizeof(float);
    float *Q = nullptr, *K = nullptr, *V = nullptr, *attn_before_out = nullptr;
    CHECK_CUDA(cudaMalloc(&Q, proj_size));
    CHECK_CUDA(cudaMalloc(&K, proj_size));
    CHECK_CUDA(cudaMalloc(&V, proj_size));
    CHECK_CUDA(cudaMalloc(&attn_before_out, proj_size));
    
    // Use cublas for linear projections.
    cublasHandle_t cublas_handle;
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetStream(cublas_handle, stream));
    
    // In projection: Q = query @ W_q^T + b_q, where W_q is first E rows of in_proj_weight.
    // Actually in_proj_weight shape is [3*E, E], so W_q = in_proj_weight[0:E, :], W_k = [E:2E, :], W_v = [2E:3E, :].
    // We'll do batched GEMM: treat B*S as batch for each projection.
    // For each projection, we need to multiply input matrix of size (B*S, E) by weight (E, E) transposed to get (B*S, E).
    // We can use cublasSgemmStridedBatched or just loop over batch dims? Better use cublasSgemm with leading dimensions.
    // We'll use cublasSgemm with column-major convention? PyTorch is row-major, so we need to be careful.
    // cublas uses column-major (Fortran), but we can treat our row-major matrices as transposed.
    // Simpler: use cublasSgemm with opA=CUBLAS_OP_T, opB=CUBLAS_OP_N.
    // For projection: output = input ^T (weights) + bias.
    // We'll do: out = input @ weight^T. In cublas, if we store input as row-major, we can call cublasSgemm with opA=CUBLAS_OP_T to interpret input^T as column-major, etc.
    // Let's do this: For each combination, we can compute using cublasSgemmStridedBatched to handle batch of matrices.
    // But to keep it simple, we can launch separate kernels for projection without cuBLAS? But requirement says use cuBLAS for GEMM.
    // We'll use cublasSgemm for a single matrix by treating B*S as one dimension.
    int M = B * S; // number of rows in input matrix (each row is an example)
    int N = E;     // output features
    int K_dim = E; // input features
    float alpha = 1.0f, beta = 0.0f;
    // For Q: weight submatrix W_q is the first E rows of in_proj_weight.
    const float* W_q = in_proj_weight;
    const float* W_k = in_proj_weight + E * E;
    const float* W_v = in_proj_weight + 2 * E * E;
    const float* b_q = in_proj_bias;
    const float* b_k = in_proj_bias + E;
    const float* b_v = in_proj_bias + 2 * E;
    
    // Compute Q = query * W_q^T + b_q
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K_dim, &alpha, W_q, K_dim, query, K_dim, &beta, Q, N);
    // Explanation: We want C = A * B, where A is query (M x K_dim), B is W_q^T (K_dim x N).
    // In column-major, we can do C^T = B^T * A^T. Or use appropriate op flags.
    // We'll use cublasSgemm with opA=CUBLAS_OP_T to transpose the weight? Actually weight is already stored as W_q (N x K_dim) and we want W_q^T (K_dim x N). So we can set opA=CUBLAS_OP_N for W_q? Let's derive.
    // We want output O = X * W^T, where X is M x K, W is N x K.
    // cublas is column-major, but if we pass X and W with opA=CUBLAS_OP_N and opB=CUBLAS_OP_T, we compute C = op(A) * op(B). If A is X stored column-major? Actually, cublasSgemm expects matrices in column-major, but our data is row-major.
    // We'll use the standard trick: treat row-major as column-major of transposed matrices.
    // Let's define: We have X (M x K) row-major. The same memory can be seen as X^T (K x M) column-major (since row-major of X is column-major of X^T). Similarly, W^T (K x N) row-major is column-major of W (N x K).
    // So we can call cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K_dim, &alpha, W, N, X, K_dim, &beta, O, N) to compute O = W * X? That's not what we want.
    // Let's use explicit transposes: O = X * W^T. In column-major, we want C^T = W * X^T? Hmm.
    // Standard approach: cublasSgemm with opA=CUBLAS_OP_N, opB=CUBLAS_OP_T: C = A * B^T, where A is stored column-major, B is stored column-major. But if our A (X) is row-major, we can pass it as A^T (column-major) with opA=CUBLAS_OP_T? Many examples do: if you have row-major matrices, you can call cublasSgemm with opA=CUBLAS_OP_T on the weight and opB=CUBLAS_OP_N on the input to compute output = input * weight^T, but with leading dimensions adjusted.
    // Let's be more concrete: We'll write a helper function to perform gemm with row-major using cublas.
    // We'll loop over batch separately? Since B*S is large (16384), we can just do one gemm with M=16384.
    // Let's use cublasSgemm with:
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K_dim, &alpha, W, N, X, K_dim, &beta, Y, N);
    // This computes Y = W * X^T, dims: (N x K) * (K x M) = N x M, but we want M x N. That's actually Y^T.
    // So we can compute Y^T = W * X^T, then our desired output is Y = (W * X^T)^T = X * W^T. That works, but we need to store Y as column-major? Actually cublas returns Y in column-major, which in memory is row-major of Y^T. So if we allocate Y with size M*N, treating it as row-major of output, cublas will fill it as if it were column-major of a N x M matrix, which is the transpose. So the memory layout will be wrong unless we account for that.
    // The common trick: if you have row-major matrices, you can call cublasSgemm with opA=CUBLAS_OP_T and opB=CUBLAS_OP_N, but swap A and B in call.
    // Let's use cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, W, K, X, K, &beta, Y, N) where W is N x K stored row-major, X is M x K stored row-major, and Y will be M x N stored row-major.
    // This is a well-known method. We'll do that.
    // So for Q: call cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, E, B*S, E, &alpha, W_q, E, query, E, &beta, Q, E);
    // But need to add bias afterwards. We'll do a separate bias add kernel or use cublasLt? Simpler: allocate Q with zero and use beta=1? We can set beta=0 for multiplication, then add bias with a custom kernel.
    // We'll write a simple kernel to add bias to each row.
    
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, E, M, E, &alpha, W_q, E, query, E, &beta, Q, E);
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, E, M, E, &alpha, W_k, E, key, E, &beta, K, E);
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, E, M, E, &alpha, W_v, E, value, E, &beta, V, E);
    
    // Add biases (all three)
    // Launch kernel to add bias to each row of Q, K, V
    int total_rows = M;
    int threads = 256;
    int blocks = (total_rows + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(Q, b_q, E, M);
    add_bias_kernel<<<blocks, threads, 0, stream>>>(K, b_k, E, M);
    add_bias_kernel<<<blocks, threads, 0, stream>>>(V, b_v, E, M);
    
    // Initialize attn_before_out to zero
    cudaMemsetAsync(attn_before_out, 0, proj_size, stream);
    
    // Launch attention kernel: grid dims (S, H, B)
    dim3 grid(S, H, B);
    // block dimension: 256 threads
    int block = 256;
    // shared memory: S scores + reduction arrays (num_warps for max, num_warps for sum + maybe padding)
    int num_warps = (block + 31) / 32;
    int shared_scores = S * sizeof(float);
    int shared_reduce = num_warps * sizeof(float) * 2; // max and sum arrays
    size_t shared_total = shared_scores + shared_reduce;
    attention_forward_kernel<<<grid, block, shared_total, stream>>>(
        Q, K, V, attn_before_out, attn_output_weights, B, S, E, H, D, scale);
    
    // After attention, attn_output_weights already contains averaged probs (since we used 1/H factor).
    
    // Out projection: attn_output = attn_before_out @ out_proj_weight^T + out_proj_bias
    // Similar gemm.
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, E, M, E, &alpha, out_proj_weight, E, attn_before_out, E, &beta, attn_output, E);
    // Add bias
    add_bias_kernel<<<blocks, threads, 0, stream>>>(attn_output, out_proj_bias, E, M);
    
    // Clean up
    cublasDestroy(cublas_handle);
    cudaFree(Q); cudaFree(K); cudaFree(V); cudaFree(attn_before_out);
    // Note: we should not return before checking errors, but we'll rely on caller to check.
}

// Bias addition kernel for GEMM output (out = in + bias for each row)
__global__ void add_bias_kernel(float* out, const float* bias, int E, int M) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M) {
        int row_start = idx * E;
        for (int i = 0; i < E; ++i) {
            out[row_start + i] += bias[i];
        }
    }
}

// ---------------------------------------------------------------------
// concat_dim2 kernel: concatenate along last dimension
// Inputs: tensor1 [B, S, E1], tensor2 [B, S, E2], output [B, S, E1+E2]
// Assume contiguous, float32.
__global__ void concat_dim2_kernel(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ out,
    int B, int S, int E1, int E2)
{
    int total = B * S;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int row = idx;
    int out_row_start = row * (E1 + E2);
    int a_row_start = row * E1;
    int b_row_start = row * E2;
    // Copy first part
    for (int i = 0; i < E1; ++i) {
        out[out_row_start + i] = a[a_row_start + i];
    }
    // Copy second part
    for (int i = 0; i < E2; ++i) {
        out[out_row_start + E1 + i] = b[b_row_start + i];
    }
}

extern "C" void concat_dim2_launcher(
    const float* a, const float* b, float* out,
    int B, int S, int E1, int E2,
    cudaStream_t stream)
{
    int total_rows = B * S;
    int threads = 256;
    int blocks = (total_rows + threads - 1) / threads;
    concat_dim2_kernel<<<blocks, threads, 0, stream>>>(a, b, out, B, S, E1, E2);
}

// ---------------------------------------------------------------------
// smooth_l1_loss kernel: compute mean
// Input: input [N], target [N], beta scalar
// Output: loss scalar
// We'll use a two-pass reduction: first compute per-element loss and sum, then divide by N.
// Each block computes partial sum, then atomically add to global accumulator, or use a second kernel.
// For simplicity, we'll use atomicAdd on a device scalar, then copy to output.
// But we need to handle N possibly large (N = B*S*2*E = 16*1024*512 = 8388608).
// We'll use a grid-stride loop and block reduction.
__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ input, const float* __restrict__ target,
    float beta, int N, float* __restrict__ partial_sums)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    // grid-stride loop
    int stride = blockDim.x * gridDim.x;
    for (int idx = i; idx < N; idx += stride) {
        float diff = input[idx] - target[idx];
        float abs_diff = fabsf(diff);
        float loss_val;
        if (abs_diff < beta) {
            loss_val = 0.5f * diff * diff / beta;
        } else {
            loss_val = abs_diff - 0.5f * beta;
        }
        sum += loss_val;
    }
    // Reduce within block
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(partial_sums, sdata[0]);
    }
}

extern "C" void smooth_l1_loss_launcher(
    const float* input, const float* target, float beta,
    int N, float* output, cudaStream_t stream)
{
    // output is pointer to a scalar float on device, initialized to 0.
    // We'll allocate a device zero buffer for partial_sums.
    cudaMemsetAsync(output, 0, sizeof(float), stream);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // cap grid size, but grid-stride loop handles rest
    if (blocks == 0) blocks = 1;
    int shared_mem = threads * sizeof(float);
    smooth_l1_loss_kernel<<<blocks, threads, shared_mem, stream>>>(input, target, beta, N, output);
    // Now output contains sum, need to divide by N to get mean.
    // We'll launch a small kernel to divide by N, or use cudaMemcpyAsync and do in second kernel.
    // Simpler: launch a single kernel to do final division.
    divide_scalar_kernel<<<1, 1, 0, stream>>>(output, (float)N);
}

__global__ void divide_scalar_kernel(float* value, float divisor) {
    *value = *value / divisor;
}
