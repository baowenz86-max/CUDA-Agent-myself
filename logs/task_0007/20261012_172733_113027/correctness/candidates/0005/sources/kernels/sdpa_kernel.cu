// Scaled dot-product attention forward: attn = softmax((Q @ K^T) * scale + attn_mask) @ V
// Q: [batch, seq_len_q, head_dim], K: [batch, seq_len_k, head_dim], V: [batch, seq_len_k, head_dim]
// attn_mask: [seq_len_q, seq_len_k] (broadcast over batch) of additive mask, typically upper triangular -inf.
// Output: [batch, seq_len_q, head_dim]
// Only float32, contiguous, CUDA.

#include <cuda_runtime.h>
#include <math.h>

// This implementation uses per-batch processing: each block handles one query position for a given batch, computes attention over all keys, then accumulates weighted V.
// This is not the most efficient for large seq_len but should be correct and reasonably fast given the dimensions (seq_len=1024, head_dim=64).
// To improve performance, we could tile over query positions with shared memory caching of K and V, but for simplicity we start with a straightforward kernel.
// We use one thread per output element (batch, q_idx, head_dim). Each thread computes the full attention row and dot product with V column.

__global__ void sdpa_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ attn_mask,
    int batch_size,
    int seq_len,
    int head_dim,
    float scale
) {
    int total = batch_size * seq_len * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        int head = i % head_dim;
        int q_idx = (i / head_dim) % seq_len;
        int batch = i / (head_dim * seq_len);

        // Compute attention scores for this query against all keys
        float max_score = -INFINITY;
        float sum_exp = 0.0f;
        float acc = 0.0f;

        // We need to store scores? We can do two passes: first find max, then compute softmax and weighted sum.
        // Since head_dim is small, we can do two passes over keys.
        // First pass: compute max score for numerical stability.
        const float* q_row = query + (batch * seq_len + q_idx) * head_dim;
        for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
            const float* k_row = key + (batch * seq_len + k_idx) * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_row[d] * k_row[d];
            }
            score *= scale;
            // Add mask
            float mask_val = attn_mask[q_idx * seq_len + k_idx]; // mask is [seq_len, seq_len]
            score += mask_val;
            if (score > max_score) max_score = score;
        }

        // Second pass: compute softmax and weighted sum with V
        for (int k_idx = 0; k_idx < seq_len; ++k_idx) {
            const float* k_row = key + (batch * seq_len + k_idx) * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += q_row[d] * k_row[d];
            }
            score = (score * scale) + attn_mask[q_idx * seq_len + k_idx];
            float exp_score = expf(score - max_score);
            sum_exp += exp_score;
            const float* v_row = value + (batch * seq_len + k_idx) * head_dim;
            acc += exp_score * v_row[head];
        }

        output[i] = acc / sum_exp;
    }
}

extern "C" void sdpa_forward_launcher(
    float* output,
    const float* query,
    const float* key,
    const float* value,
    const float* attn_mask,
    int batch_size,
    int seq_len,
    int head_dim,
    cudaStream_t stream
) {
    float scale = rsqrtf((float)head_dim); // 1/sqrt(head_dim)
    int total = batch_size * seq_len * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    sdpa_forward_kernel<<<blocks, threads, 0, stream>>>(
        output, query, key, value, attn_mask, batch_size, seq_len, head_dim, scale);
}
