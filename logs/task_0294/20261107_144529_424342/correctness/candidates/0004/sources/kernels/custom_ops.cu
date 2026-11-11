#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel for embedding lookup (forward)
// indices: int64_t array of shape [total_indices]
// weight: float array of shape [num_embeddings, embedding_dim], row-major
// output: float array of shape [total_indices, embedding_dim]
// For each i in [0,total_indices): row_idx = indices[i];
// if row_idx == padding_idx: fill output row with 0
// else copy weight[row_idx, :] to output row
// Supports arbitrary weight strides: weight_row_stride is number of elements between rows,
// weight_col_stride is number of elements between columns.
// For contiguous weight (col_stride=1), we use simple copy loop; for non-contiguous, we use strided copy.
__global__ void embedding_lookup_forward_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t total_indices,
    int embedding_dim,
    int weight_row_stride,
    int weight_col_stride,
    int64_t padding_idx)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (; idx < total_indices; idx += stride) {
        int64_t row_idx = indices[idx];
        float* out_row = output + idx * embedding_dim;

        if (row_idx == padding_idx) {
            // Zero this row
            for (int j = 0; j < embedding_dim; ++j) {
                out_row[j] = 0.0f;
            }
        } else {
            // Valid row: copy from weight
            const float* weight_row = weight + row_idx * weight_row_stride;
            if (weight_col_stride == 1) {
                // Contiguous row in weight, can copy directly
                #pragma unroll 4
                for (int j = 0; j < embedding_dim; ++j) {
                    out_row[j] = weight_row[j];
                }
            } else {
                // Non-contiguous columns: use stride
                for (int j = 0; j < embedding_dim; ++j) {
                    out_row[j] = weight_row[j * weight_col_stride];
                }
            }
        }
    }
}

// Host launcher for embedding lookup forward
// Parameters:
//   indices: pointer to int64 indices on device, shape [total_indices]
//   weight: pointer to float weight on device, shape [num_embeddings, embedding_dim]
//   output: pointer to float output on device, shape [total_indices, embedding_dim]
//   total_indices: number of indices = batch_size * seq_length
//   embedding_dim: dimension of each embedding vector
//   weight_row_stride: stride between consecutive rows in weight (elements), usually embedding_dim
//   weight_col_stride: stride between consecutive columns in weight (elements), usually 1
//   padding_idx: index to treat as padding; output rows for this index will be zero
//   stream: CUDA stream to execute on
extern "C" void embedding_lookup_forward_launcher(
    const int64_t* indices,
    const float* weight,
    float* output,
    int64_t total_indices,
    int embedding_dim,
    int weight_row_stride,
    int weight_col_stride,
    int64_t padding_idx,
    cudaStream_t stream)
{
    // Determine block size and grid size
    int block_size = 256;
    int64_t grid_size = (total_indices + block_size - 1) / block_size;
    // Cap grid size if needed (avoid exceeding max grid dimension)
    if (grid_size > 1048576) {
        grid_size = 1048576;
    }

    embedding_lookup_forward_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        indices, weight, output, total_indices, embedding_dim,
        weight_row_stride, weight_col_stride, padding_idx);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error will be caught by binding after launch; nothing to do here
    }
}
