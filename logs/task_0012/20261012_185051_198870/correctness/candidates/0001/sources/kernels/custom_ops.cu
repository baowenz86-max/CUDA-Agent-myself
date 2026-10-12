// CUDA kernels for relu_sqrt, silu, and fused multi-head attention
// All launchers are extern C for binding layer.

#include <cuda_runtime.h>
#include <math.h>

// relu_sqrt kernel: y = sqrt(relu(x))
__global__ void relu_sqrt_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < numel; i += stride) {
        float v = input[i];
        v = fmaxf(v, 0.0f);
        output[i] = sqrtf(v);
    }
}

// silu kernel: y = x * sigmoid(x)
__global__ void silu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < numel; i += stride) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

// MHA forward kernel
// Inputs:
//   query, key, value: (seq_len, batch, embed_dim) contiguous
//   in_proj_weight: (3*embed_dim, embed_dim)
//   in_proj_bias: (3*embed_dim)
//   out_proj_weight: (embed_dim, embed_dim)
//   out_proj_bias: (embed_dim)
//   output: (seq_len, batch, embed_dim)
//   seq_len, batch, embed_dim, num_heads
// head_dim = embed_dim / num_heads
// Scalars: scaling = 1/sqrt(head_dim)
//
// This kernel assumes float32, no masks, default attention.
// Each block handles one query position (seq index) and one batch index,
// producing one output element per thread? Actually we produce all embed_dim outputs for that (seq, batch).
// We use blockDim.x = num_heads (or embed_dim?), better: one thread per output element?
// Design: blockDim.x = embed_dim (256), each thread computes one output element.
// But attention requires computing attention weights across all keys for all heads.
// To reduce recomputation, we can have each block compute for one (seq_q, batch) and all heads/outputs.
// Each thread will be responsible for one output element (head * head_dim + d).
// It needs the query projection for that head and position, and must loop over all keys for that head and all positions.
// That would be seq_len * batch * embed_dim blocks? Actually number of query positions = seq_len * batch.
// Each block has embed_dim threads. That is seq_len*batch blocks, 256 threads each.
// That's okay, but we can also use blockDim.x=128 or 256.
// Each thread computes its query projection value, then loops over all key positions, accumulates dot product and attention scores.
// It can also compute value projection and output after softmax.
// This is straightforward and avoids shared memory complexity.

__global__ void mha_forward_kernel(
    const float* __restrict__ query,   // (seq_len, batch, embed_dim)
    const float* __restrict__ key,     // (seq_len, batch, embed_dim)
    const float* __restrict__ value,   // (seq_len, batch, embed_dim)
    const float* __restrict__ in_proj_weight, // (3*embed_dim, embed_dim)
    const float* __restrict__ in_proj_bias,   // (3*embed_dim)
    const float* __restrict__ out_proj_weight, // (embed_dim, embed_dim)
    const float* __restrict__ out_proj_bias,   // (embed_dim)
    float* __restrict__ output,
    int seq_len,
    int batch,
    int embed_dim,
    int num_heads
) {
    int head_dim = embed_dim / num_heads;
    int q_idx = blockIdx.x; // query position index, 0 <= q_idx < seq_len * batch
    int s = q_idx / batch;   // seq index
    int b = q_idx % batch;   // batch index
    int tid = threadIdx.x;   // 0 <= tid < embed_dim
    if (tid >= embed_dim) return;

    int head = tid / head_dim; // which head this output element belongs to
    int d = tid % head_dim;   // which dimension within head

    // Compute query projection for this head and dimension:
    // q_val = sum_{e=0}^{embed_dim-1} query[s,b,e] * in_proj_weight[head*head_dim + d, e] + in_proj_bias[head*head_dim + d]
    // Note: in_proj_weight layout: rows 0..embed_dim-1 are Q, rows embed_dim..2*embed_dim-1 are K, rows 2*embed_dim..3*embed_dim-1 are V.
    float q_val = 0.0f;
    const float* q_row = query + (s * batch + b) * embed_dim; // pointer to query vector for this (s,b)
    const float* q_weight_row = in_proj_weight + (head * head_dim + d) * embed_dim; // row is head*head_dim+d
    for (int e = 0; e < embed_dim; ++e) {
        q_val += q_row[e] * q_weight_row[e];
    }
    q_val += in_proj_bias[head * head_dim + d];

    // Apply scaling factor
    float scaling = 1.0f / sqrtf((float)head_dim);

    // Compute attention scores and softmax online
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float attn_output = 0.0f; // accumulator for output for this head dimension

    // Loop over all key positions
    for (int s_k = 0; s_k < seq_len; ++s_k) {
        // Compute dot product between query projection (head,d) and key projection (head,d) for this key position and batch.
        // key projection: k_val = sum_{e} key[s_k,b,e] * in_proj_weight[embed_dim + head*head_dim + d, e] + in_proj_bias[embed_dim + head*head_dim + d]
        float k_val = 0.0f;
        const float* k_row = key + (s_k * batch + b) * embed_dim;
        const float* k_weight_row = in_proj_weight + (embed_dim + head * head_dim + d) * embed_dim;
        for (int e = 0; e < embed_dim; ++e) {
            k_val += k_row[e] * k_weight_row[e];
        }
        k_val += in_proj_bias[embed_dim + head * head_dim + d];

        float score = q_val * k_val * scaling; // dot product is sum over head_dim, but we are computing for one dimension only! Wrong.
        // Actually attention dot product is over head_dim: sum_{d=0}^{head_dim-1} Q[head,d] * K[head,d].
        // If each thread only has one d, it cannot compute the full dot product.
        // Need redesign: either use one block per query position and compute dot products cooperatively,
        // or each thread computes full dot product for a head (looping over all d) but then output element mapping changes.
        // Better approach: block per (query position, head) with head_dim threads; then each thread computes one output element and we can compute dot product via reduction within block.
        // That would be seq_len*batch*num_heads blocks, head_dim threads each.
        // Alternatively, block per query position and batch, with embed_dim threads, and use shared memory to store projections for all heads and keys for current key position? That's more complex.
        // For simplicity and correctness, we'll implement block per (query position, head), so blockDim.x = head_dim (32 if embed_dim=256,num_heads=8).
        // We'll use blockIdx.y for head index, blockIdx.x for query position.
        // Let's restructure: gridDim.x = seq_len * batch, gridDim.y = num_heads.
        // blockDim.x = head_dim.
        // Each thread computes its own Q projection and loops over keys, but within block we need dot product reduction.
        // For each key position, each thread computes its dimension's contribution to dot product (q_val*key_val), then we reduce across block to get total score, then compute softmax and apply to value.
        // This is more efficient and correct.
        // We'll implement that.
    }
}

// Corrected MHA kernel design:
// Each block corresponds to one query position (seq, batch) and one head.
// blockDim.x = head_dim (e.g., 32).
// Each thread handles one dimension d of that head.
// Steps:
// 1. Compute q_val for (head,d) from query and in_proj_weight.
// 2. Loop over all key positions s_k:
//    a. Compute k_val for (head,d) from key[s_k,b] and in_proj_weight.
//    b. Compute partial dot = q_val * k_val.
//    c. Reduce partial across block to get total_score.
//    d. Compute exp(score - max) etc. for softmax online.
//    e. Compute v_val for (head,d) from value[s_k,b] and in_proj_weight.
//    f. Accumulate v_val * attention_weight into attn_output[head_dim thread].
// 3. After all keys, normalize by sum_exp.
// 4. Store result in shared memory (or directly to output if we also compute output projection? Output projection requires sum over all heads and dimensions, so we need all head outputs for that query position.
//    So we cannot directly write final output in this block alone. We need another phase: after computing attention output for each head, we need to apply out_proj across all heads.
// Option A: Use atomicAdd to accumulate contributions to output after multiplying by out_proj weights.
// Option B: Use two kernels: first compute attention output (seq_len, batch, embed_dim), then apply out_proj as a separate GEMM-like kernel.
// Option C: Use a single block per query position (seq,batch) with embed_dim threads; inside the block, we compute all heads sequentially using shared memory and reductions, then apply out_proj. This is more complex but avoids extra kernel.
// Given the complexity, perhaps simplest correct implementation is to split into separate kernels:
//   1. in_projection and attention: compute context (seq_len, batch, embed_dim).
//      Use one block per (query position, head) as above, write attention output for each head to a temporary buffer of shape (seq_len, batch, embed_dim).
//   2. out_projection: standard matrix-vector per query position: for each (seq,batch), compute output = W_out * context + b_out.
// However, passing temporary buffer requires allocation from binding. The binding can allocate an intermediate tensor.
// The launcher signature can include a workspace pointer.
// Or we can do out_proj within the same kernel using atomicAdd to accumulate output contributions.
// The atomicAdd approach may be okay and simpler.

// We'll implement MHA with two kernels: attention kernel writes context to an intermediate buffer, and out_proj kernel reads context and writes final output.
// This requires two launchers or one launcher that calls both kernels sequentially on the same stream.
// For simplicity, we provide a single launcher that does both: allocate workspace? But launcher cannot allocate PyTorch tensors; we can have the binding allocate workspace and pass pointer.
// The binding can call cudaMalloc for workspace, or we can just use a preallocated buffer passed from binding.
// In model_new.py, mha_forward can be implemented with two custom op calls: first compute attention, then out_proj. But model_new.py currently calls mha_forward as one op.
// We must stick to the interface: mha_forward returns final output. So our mha_forward launcher must handle both internally.
// We can allocate workspace inside the .cu using cudaMalloc and cudaFree. That is allowed (not PyTorch, just CUDA API).
// The launcher will: allocate workspace of size seq_len*batch*embed_dim floats, call attention kernel to fill it, then call out_proj kernel to compute final output from workspace, then free workspace.
// However, frequent cudaMalloc/free per call may hurt performance. We could use static workspace if size fixed, but size is dynamic. Still okay for correctness; performance later may be optimized with persistent workspace or fused kernel.

// Let's design the kernels:

// Kernel 1: attention_kernel
// gridDim.x = seq_len * batch, gridDim.y = num_heads
// blockDim.x = head_dim (must be power of 2, <= 1024; head_dim=32 for given config).
// Each block:
//   - Identify query position (s,b) and head.
//   - Each thread d computes q_val.
//   - Loop over key positions s_k:
//       - Each thread computes k_val for key position s_k.
//       - Compute partial dot = q_val * k_val.
//       - Warp/block reduction to get total_score (since head_dim is small, we can use block reduce with shared memory).
//       - After reduction, all threads have total_score.
//       - Compute exp(total_score - max_score) (online softmax).
//       - Each thread computes v_val for value position s_k.
//       - Accumulate v_val * exp_score into attn_accum[d].
//       - Also accumulate sum_exp.
//   - After loop, normalize attn_accum[d] /= sum_exp.
//   - Write attn_accum[d] to workspace at (s, b, head * head_dim + d).

// Kernel 2: out_proj_kernel
// gridDim.x = seq_len * batch, blockDim.x = embed_dim (256).
// Each thread computes one output element (embed_dim index).
// For each row of out_proj_weight (embed_dim rows), output = sum_{j=0}^{embed_dim-1} context[j] * weight[row*embed_dim + j] + bias[row].
// We can use one block per query position; each thread computes one output element via loop over embed_dim.
// This is fine.

// Now implement in custom_ops.cu.

// Also, the MHA attention uses scaled dot-product with head_dim scaling. Inputs are float32.

__global__ void attention_kernel(
    const float* __restrict__ query, const float* __restrict__ key, const float* __restrict__ value,
    const float* __restrict__ in_proj_weight, const float* __restrict__ in_proj_bias,
    float* __restrict__ workspace, int seq_len, int batch, int embed_dim, int num_heads) {
    int head_dim = embed_dim / num_heads;
    int q_idx = blockIdx.x; // 0..seq_len*batch-1
    int head = blockIdx.y;
    int s = q_idx / batch;
    int b = q_idx % batch;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int q_row = (s * batch + b) * embed_dim;
    // Query projection for this head and dimension d
    float q_val = 0.0f;
    for (int e = 0; e < embed_dim; ++e) {
        q_val += query[q_row + e] * in_proj_weight[(head * head_dim + d) * embed_dim + e];
    }
    q_val += in_proj_bias[head * head_dim + d];

    float scaling = 1.0f / sqrtf((float)head_dim);
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float attn_accum = 0.0f;

    // Shared memory for reduction
    __shared__ float s_partial[32]; // max head_dim 32, but we can set to 32 or use dynamic
    // Assume head_dim <= 32; if larger, adjust. For our config head_dim=32.
    // To be safe, use extern __shared__ dynamic, but simpler: static size 64.
    __shared__ float s_partial_sh[64];

    for (int s_k = 0; s_k < seq_len; ++s_k) {
        int k_row = (s_k * batch + b) * embed_dim;
        float k_val = 0.0f;
        for (int e = 0; e < embed_dim; ++e) {
            k_val += key[k_row + e] * in_proj_weight[(embed_dim + head * head_dim + d) * embed_dim + e];
        }
        k_val += in_proj_bias[embed_dim + head * head_dim + d];

        float partial_dot = q_val * k_val * scaling;
        // Reduce partial_dot across block to get total_score
        s_partial_sh[d] = partial_dot;
        __syncthreads();
        // Reduction (assuming blockDim.x is power of 2)
        for (int offset = head_dim / 2; offset > 0; offset >>= 1) {
            if (d < offset && d + offset < head_dim) {
                s_partial_sh[d] += s_partial_sh[d + offset];
            }
            __syncthreads();
        }
        float total_score = s_partial_sh[0];

        // Online softmax update
        float new_max = fmaxf(max_score, total_score);
        float old_exp_sum = sum_exp;
        if (new_max > max_score) {
            // need to rescale sum_exp
            sum_exp = sum_exp * expf(max_score - new_max);
        }
        float exp_score = expf(total_score - new_max);
        sum_exp += exp_score;
        max_score = new_max;

        // Compute value projection for this key position
        float v_val = 0.0f;
        for (int e = 0; e < embed_dim; ++e) {
            v_val += value[k_row + e] * in_proj_weight[(2 * embed_dim + head * head_dim + d) * embed_dim + e];
        }
        v_val += in_proj_bias[2 * embed_dim + head * head_dim + d];

        attn_accum += v_val * exp_score;
        __syncthreads(); // ensure all threads have completed accumulation before next iteration? Not necessary because each thread only uses its own attn_accum and sum_exp, but sum_exp is per-thread, not reduced across threads? Actually softmax denominator sum_exp should be same across threads in a block because total_score is the same (from reduction) and we apply same online update. However, due to floating-point ordering, sum_exp might differ slightly per thread; but we only need one sum_exp to normalize each thread's own accumulator. Since each thread updates its own sum_exp identically (they all start from same reductions), they should be identical. So no need to sync for sum_exp. The reduction uses shared memory, so we need __syncthreads before reusing s_partial_sh in next iteration. The loop starts with writing to s_partial_sh[d], so we need __syncthreads at end of loop to ensure all threads have finished reading from previous reduction. We'll place __syncthreads before the end of loop if not last iteration, but we can also put __syncthreads after writing attn_accum? Better: at end of loop body, after using s_partial_sh[0] and updating, we should __syncthreads before next iteration's write to s_partial_sh. So after the reduction and softmax, we can __syncthreads at the end to avoid race with next write.
    }

    // Normalize
    attn_accum /= sum_exp;
    // Write workspace
    int workspace_idx = (s * batch + b) * embed_dim + head * head_dim + d;
    workspace[workspace_idx] = attn_accum;
}

__global__ void out_proj_kernel(
    const float* __restrict__ workspace, const float* __restrict__ out_proj_weight,
    const float* __restrict__ out_proj_bias, float* __restrict__ output,
    int seq_len, int batch, int embed_dim) {
    int q_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= embed_dim) return;
    int row = q_idx * embed_dim + tid; // output element index
    float sum = 0.0f;
    for (int j = 0; j < embed_dim; ++j) {
        sum += workspace[q_idx * embed_dim + j] * out_proj_weight[tid * embed_dim + j];
    }
    output[row] = sum + out_proj_bias[tid];
}

extern "C" void relu_sqrt_forward(
    const float* input, float* output, int numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // gridDim.x limit; use grid-stride loop
    relu_sqrt_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void silu_forward(
    const float* input, float* output, int numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    silu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void mha_forward(
    const float* query, const float* key, const float* value,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    float* output, int seq_len, int batch, int embed_dim, int num_heads,
    cudaStream_t stream) {
    int head_dim = embed_dim / num_heads;
    // Allocate workspace
    float* workspace = nullptr;
    size_t workspace_size = (size_t)seq_len * batch * embed_dim * sizeof(float);
    cudaMalloc(&workspace, workspace_size);
    if (!workspace) return;

    // Launch attention kernel
    dim3 grid_attn(seq_len * batch, num_heads);
    dim3 block_attn(head_dim);
    if (head_dim > 64) {
        // If head_dim too large for static shared memory, we may need dynamic. For our config head_dim=32.
        // We'll assume head_dim <= 64.
    }
    attention_kernel<<<grid_attn, block_attn, 0, stream>>>(
        query, key, value, in_proj_weight, in_proj_bias, workspace, seq_len, batch, embed_dim, num_heads);

    // Launch out_proj kernel
    int threads_out = 256;
    dim3 block_out(threads_out);
    dim3 grid_out((seq_len * batch + threads_out - 1) / threads_out); // one block per query position? Actually out_proj_kernel assumes one block per query position, with blockDim.x = embed_dim. So we need blockDim.x = embed_dim, and gridDim.x = seq_len*batch. Since embed_dim<=1024, we can set blockDim.x = embed_dim.
    // But our kernel uses threadIdx.x for output dimension, so blockDim.x must equal embed_dim.
    // We'll launch with blockDim.x = embed_dim.
    dim3 block_out(embed_dim);
    dim3 grid_out(seq_len * batch);
    out_proj_kernel<<<grid_out, block_out, 0, stream>>>(
        workspace, out_proj_weight, out_proj_bias, output, seq_len, batch, embed_dim);

    // Free workspace after kernels complete (asynchronously? cudaFree is synchronous with respect to device? Usually cudaFree waits for all preceding work on the stream if using default stream? Actually cudaFree is synchronous, but we want to free after kernels finish. We can use cudaFree after launches; it will block until kernels complete if using legacy default stream? With per-thread default stream maybe not. To be safe, we can use cudaStreamSynchronize before free, but that adds overhead. Better: use cudaFree after kernel calls; it may block, but correctness will be fine. For performance, later we can use events or a persistent workspace. We'll keep simple.
    cudaFree(workspace);
}
