// kernels/custom_ops.cu
// CUDA kernels and launchers for embedding_add_forward
#include <cuda_runtime.h>

// Kernel: compute output[b,s,:] = weight1[x1[b,s], :] + weight2[x2[b,s], :]
// Inputs x1, x2 are int64 index tensors with shape [batch, seq_len]
// weight1: [num_embeddings_1, embedding_dim]
// weight2: [num_embeddings_2, embedding_dim]
// Output: [batch, seq_len, embedding_dim] float32
// All pointers are raw device pointers.
__global__ void embedding_add_kernel(
    const int64_t* __restrict__ x1,
    const int64_t* __restrict__ x2,
    const float* __restrict__ weight1,
    const float* __restrict__ weight2,
    float* __restrict__ output,
    int64_t num_embeddings_1,
    int64_t num_embeddings_2,
    int64_t embedding_dim,
    int64_t batch,
    int64_t seq_len,
    int64_t stride_x1_batch,
    int64_t stride_x1_seq,
    int64_t stride_x2_batch,
    int64_t stride_x2_seq)
{
    // Total number of output rows = batch * seq_len
    int64_t total_rows = batch * seq_len;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_rows) return;

    // Decode (b, s) from linear index
    int64_t b = idx / seq_len;
    int64_t s = idx - b * seq_len;

    // Load indices (with strides)
    int64_t i1 = x1[b * stride_x1_batch + s * stride_x1_seq];
    int64_t i2 = x2[b * stride_x2_batch + s * stride_x2_seq];

    // Assume indices are in range (contract says valid)
    // Compute output row pointer: output + idx * embedding_dim
    float* out_row = output + idx * embedding_dim;
    const float* w1_row = weight1 + i1 * embedding_dim;
    const float* w2_row = weight2 + i2 * embedding_dim;

    // Elementwise add over embedding_dim
    for (int64_t e = 0; e < embedding_dim; ++e) {
        out_row[e] = w1_row[e] + w2_row[e];
    }
}

// Launcher function: no PyTorch dependencies
extern "C" void embedding_add_launcher(
    const int64_t* x1,
    const int64_t* x2,
    const float* weight1,
    const float* weight2,
    float* output,
    int64_t num_embeddings_1,
    int64_t num_embeddings_2,
    int64_t embedding_dim,
    int64_t batch,
    int64_t seq_len,
    int64_t stride_x1_batch,
    int64_t stride_x1_seq,
    int64_t stride_x2_batch,
    int64_t stride_x2_seq,
    cudaStream_t stream)
{
    int64_t total_rows = batch * seq_len;
    if (total_rows <= 0) return; // no work

    // Launch configuration: one thread per output row
    const int threads = 256;
    int64_t blocks = (total_rows + threads - 1) / threads;
    // Cap blocks to avoid grid dimension overflow? For large batch, we use grid stride loop or just rely on large grid.
    // CUDA grid dimension can be up to 2^31-1, so no overflow for our sizes.
    if (blocks > 2147483647) blocks = 2147483647; // safety, but unlikely

    embedding_add_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
        x1, x2, weight1, weight2, output,
        num_embeddings_1, num_embeddings_2, embedding_dim,
        batch, seq_len,
        stride_x1_batch, stride_x1_seq,
        stride_x2_batch, stride_x2_seq);
}
