#include <cuda_runtime.h>

// CUDA kernel for embedding lookup and division by 2.0
// indices: int64_t*  (CPU->GPU copied by binding)
// weight: const float*  (CUDA float32)
// output: float*  (CUDA float32)
// total_indices: number of indices = batch_size * seq_length
// embedding_dim: dimension of each embedding vector
__global__ void embedding_div_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int total_indices,
    int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_indices; i += stride) {
        int64_t token = indices[i];
        // The reference embedding uses nn.Embedding without padding_idx, so all tokens
        // are in [0, num_embeddings-1]. We assume valid indices, but add a guard to
        // prevent out-of-bounds reads; in practice indices are valid.
        if (token < 0) token = 0;
        // The weight size is [num_embeddings, embedding_dim]; num_embeddings not needed.
        // Access row token * embedding_dim.
        const float* row = weight + token * embedding_dim;
        float* out_row = output + i * embedding_dim;
        for (int j = 0; j < embedding_dim; ++j) {
            out_row[j] = row[j] / 2.0f;
        }
    }
}

// Launcher function (C linkage for binding)
// Arguments:
//   indices_ptr: pointer to int64 indices (already copied to GPU by binding)
//   weight_ptr: pointer to float weight (contiguous)
//   output_ptr: pointer to preallocated output (float32)
//   total_indices: number of indices
//   embedding_dim: dimension of embedding vectors
//   stream: CUDA stream
// Note: Binding must allocate output with size total_indices * embedding_dim * sizeof(float)
//       and copy indices to GPU before calling this.
extern "C" void embedding_div_launcher(
    const int64_t* indices_ptr,
    const float* weight_ptr,
    float* output_ptr,
    int total_indices,
    int embedding_dim,
    cudaStream_t stream)
{
    if (total_indices <= 0 || embedding_dim <= 0) return;
    const int threads = 256;
    const int blocks = (total_indices + threads - 1) / threads;
    // Grid-stride loop handles any size; we can also use a fixed grid cap for large inputs.
    // For total_indices up to 524288, blocks=2048, which is within limits.
    embedding_div_kernel<<<blocks, threads, 0, stream>>>(
        indices_ptr, weight_ptr, output_ptr, total_indices, embedding_dim);
}
