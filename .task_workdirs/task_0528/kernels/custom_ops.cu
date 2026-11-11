#include <cuda_runtime.h>
#include <cstdint>

// Embedding forward kernel
// input: int64_t pointer, shape [batch, seq_len], arbitrary strides
// weight: float pointer, shape [num_embeddings, embedding_dim], arbitrary strides
// output: float pointer, shape [batch, seq_len, embedding_dim], contiguous (as allocated)
// batch: number of rows in input
// seq_len: number of columns in input
// embedding_dim: number of columns in weight
// padding_idx: if >=0, indices equal to padding_idx produce zero embedding
// input is assumed to be on device, no CPU pointers
template <int BLOCK_SIZE>
__global__ void embedding_forward_kernel(
    const int64_t* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t batch,
    int64_t seq_len,
    int64_t embedding_dim,
    int64_t num_embeddings,
    int64_t padding_idx,
    int64_t input_stride_batch,
    int64_t input_stride_seq,
    int64_t weight_stride_row,
    int64_t weight_stride_col
) {
    // Each thread handles one element of output, but we use grid-stride loop over all output elements.
    // output shape: [batch, seq_len, embedding_dim], contiguous in row-major (batch, seq_len, embedding_dim)
    int64_t total_elements = batch * seq_len * embedding_dim;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += (int64_t)gridDim.x * blockDim.x) {
        int64_t k = idx % embedding_dim;
        int64_t tmp = idx / embedding_dim;
        int64_t j = tmp % seq_len;
        int64_t i = tmp / seq_len;

        // Input index: input[i, j]
        int64_t input_index = input[i * input_stride_batch + j * input_stride_seq];

        float val = 0.0f;
        if (padding_idx >= 0 && input_index == padding_idx) {
            val = 0.0f;
        } else {
            // Check bounds (should be valid)
            val = weight[input_index * weight_stride_row + k * weight_stride_col];
        }
        output[idx] = val;
    }
}

// LogSoftmax along specified dimension (dim = 1 for this task)
// input shape [batch, seq_len, embedding_dim] (contiguous or strided, but we require contiguity for simplicity?)
// We'll implement with support for strided input, but output contiguous.
// Actually contract says input may support strides, so we need to handle non-contiguous.
// We'll launch a kernel per embedding_dim? Better to use a two-pass approach using shared memory.
// For each batch i and embedding index k, we reduce over seq_len j.
// We can assign one block per (i, k) to compute max and sum over seq_len.
// seq_len = up to 2048, embedding_dim = 300, batch = 16
// Number of blocks = batch * embedding_dim = 16*300 = 4800, each block has threads = seq_len (maybe >1024, so need grid-stride within block or use multiple blocks with reduction).
// Simpler: use one thread per (i,k) and loop over seq_len sequentially? That would be slow but acceptable for small sizes.
// Better: use block per (i,k), with blockDim = 256 or 512 and grid-stride loop over seq_len to compute max and sum. Then second pass to compute output.
// We'll implement a kernel with two steps:
// Step 1: compute max per (batch, embedding) along seq_len
// Step 2: compute sum of exp(x - max), then output log_softmax
// To avoid separate kernels, do online softmax algorithm in one kernel? Could be complex. We'll do two kernels, but we can launch them separately from launcher.
// However, we can also compute log_softmax using a single kernel with block per (batch, embedding) and shared memory reduction.
// Let's implement a kernel that computes for each (i, k):
//   - find max over j
//   - compute sum exp(x - max)
//   - write output for all j
// We'll use a single block per (i,k), with shared memory arrays for partial max and sum.
// Block size = min(seq_len, 1024). If seq_len > 1024, use grid-stride loop within block.
// We'll also handle non-contiguous input via stride parameters.

// One block per (batch, embedding) pair.
// blockDim.x = block_size (could be 256 or 512)
// gridDim.x = batch * embedding_dim (but careful with max grid dimension in x: usually 2^31-1, so fine)
// We'll use template parameter for block size for tuning.

template <int BLOCK_SIZE>
__global__ void log_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t batch,
    int64_t seq_len,
    int64_t embedding_dim,
    int64_t input_stride_batch,
    int64_t input_stride_seq,
    int64_t input_stride_embed,
    int64_t output_stride_batch,
    int64_t output_stride_seq,
    int64_t output_stride_embed
) {
    // Determine batch index i and embedding index k from blockIdx.x
    int64_t block_id = blockIdx.x;
    int64_t k = block_id % embedding_dim;
    int64_t i = block_id / embedding_dim;

    // Shared memory for max and sum
    __shared__ float s_max;
    __shared__ float s_sum;

    if (threadIdx.x == 0) {
        s_max = -INFINITY;
        s_sum = 0.0f;
    }
    __syncthreads();

    // First pass: compute max over seq_len
    float thread_max = -INFINITY;
    for (int64_t j = threadIdx.x; j < seq_len; j += BLOCK_SIZE) {
        float val = input[i * input_stride_batch + j * input_stride_seq + k * input_stride_embed];
        thread_max = fmaxf(thread_max, val);
    }

    // Reduce thread_max across block
    // Use shared memory reduction
    __shared__ float s_partial_max[BLOCK_SIZE];
    s_partial_max[threadIdx.x] = thread_max;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_partial_max[threadIdx.x] = fmaxf(s_partial_max[threadIdx.x], s_partial_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        s_max = s_partial_max[0];
    }
    __syncthreads();

    float block_max = s_max;

    // Second pass: compute sum of exp(x - max)
    float thread_sum = 0.0f;
    for (int64_t j = threadIdx.x; j < seq_len; j += BLOCK_SIZE) {
        float val = input[i * input_stride_batch + j * input_stride_seq + k * input_stride_embed];
        thread_sum += expf(val - block_max);
    }

    __shared__ float s_partial_sum[BLOCK_SIZE];
    s_partial_sum[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_partial_sum[threadIdx.x] += s_partial_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        s_sum = s_partial_sum[0];
    }
    __syncthreads();

    float log_sum = logf(s_sum);

    // Third pass: write output log_softmax
    for (int64_t j = threadIdx.x; j < seq_len; j += BLOCK_SIZE) {
        float val = input[i * input_stride_batch + j * input_stride_seq + k * input_stride_embed];
        float log_softmax_val = val - block_max - log_sum;
        output[i * output_stride_batch + j * output_stride_seq + k * output_stride_embed] = log_softmax_val;
    }
}

// Launcher functions
extern "C" void embedding_forward_launcher(
    const int64_t* input,
    const float* weight,
    float* output,
    int64_t batch,
    int64_t seq_len,
    int64_t embedding_dim,
    int64_t num_embeddings,
    int64_t padding_idx,
    int64_t input_stride_batch,
    int64_t input_stride_seq,
    int64_t weight_stride_row,
    int64_t weight_stride_col,
    cudaStream_t stream
) {
    const int block_size = 256;
    int64_t total_elements = batch * seq_len * embedding_dim;
    int64_t blocks = (total_elements + block_size - 1) / block_size;
    // Cap grid size to avoid too many blocks? For total_elements up to ~10M, blocks ~40k, fine.
    // Use grid-stride loop anyway.
    embedding_forward_kernel<block_size><<<blocks, block_size, 0, stream>>>(
        input, weight, output, batch, seq_len, embedding_dim, num_embeddings, padding_idx,
        input_stride_batch, input_stride_seq, weight_stride_row, weight_stride_col
    );
}

extern "C" void log_softmax_forward_launcher(
    const float* input,
    float* output,
    int64_t batch,
    int64_t seq_len,
    int64_t embedding_dim,
    int64_t input_stride_batch,
    int64_t input_stride_seq,
    int64_t input_stride_embed,
    int64_t output_stride_batch,
    int64_t output_stride_seq,
    int64_t output_stride_embed,
    cudaStream_t stream
) {
    // Input and output are assumed to have same shape and strides? Output usually contiguous but we pass strides.
    // We'll use a fixed block size of 512. Ensure block size <= 1024.
    const int block_size = 512;
    int64_t blocks = batch * embedding_dim;
    dim3 grid(blocks);
    dim3 block(block_size);
    log_softmax_kernel<block_size><<<grid, block, 0, stream>>>(
        input, output, batch, seq_len, embedding_dim,
        input_stride_batch, input_stride_seq, input_stride_embed,
        output_stride_batch, output_stride_seq, output_stride_embed
    );
}
