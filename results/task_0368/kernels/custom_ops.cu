#include <cuda_runtime.h>
#include <cstdint>

// embedding_forward kernel
// input: int64 indices, shape [batch, seq]
// weight: float32, shape [num_embeddings, embedding_dim]
// output: float32, shape [batch, seq, embedding_dim]
// Each thread handles one input index and copies embedding_dim floats.
__global__ void embedding_kernel(
    const int64_t* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int total_indices,
    int embedding_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_indices; i += stride)
    {
        int64_t index = input[i];
        // assume index is valid (< num_embeddings)
        const float* weight_row = weight + index * embedding_dim;
        float* out_row = output + (int64_t)i * embedding_dim;
        for (int j = 0; j < embedding_dim; ++j)
        {
            out_row[j] = weight_row[j];
        }
    }
}

// log_softmax_forward kernel along dim=1 (seq dimension)
// input: float32, shape [batch, seq, embedding_dim]
// output: float32, same shape
// dim: must be 1
// We process one (batch, embedding) column per block. Each block reduces along seq.
// Block size is chosen as 256, and seq length is 1024.
__global__ void log_softmax_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int seq,
    int embedding_dim)
{
    // blockIdx.x = batch * embedding_dim + embedding_idx
    int block_col = blockIdx.x;
    int b = block_col / embedding_dim;
    int e = block_col % embedding_dim;

    // shared memory for max and sum
    __shared__ float smax;
    __shared__ float ssum;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // step 1: find max along seq for this (b,e)
    float local_max = -1e30f;
    for (int s = tid; s < seq; s += nthreads)
    {
        float val = input[(b * seq + s) * embedding_dim + e];
        if (val > local_max) local_max = val;
    }
    // block reduction to find max
    __shared__ float smem_max[256];
    smem_max[tid] = local_max;
    __syncthreads();
    for (int offset = nthreads / 2; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            float other = smem_max[tid + offset];
            if (other > smem_max[tid]) smem_max[tid] = other;
        }
        __syncthreads();
    }
    if (tid == 0) smax = smem_max[0];
    __syncthreads();

    float max_val = smax;

    // step 2: compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int s = tid; s < seq; s += nthreads)
    {
        float val = input[(b * seq + s) * embedding_dim + e];
        local_sum += __expf(val - max_val);
    }
    __shared__ float smem_sum[256];
    smem_sum[tid] = local_sum;
    __syncthreads();
    for (int offset = nthreads / 2; offset > 0; offset >>= 1)
    {
        if (tid < offset)
        {
            smem_sum[tid] += smem_sum[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) ssum = smem_sum[0];
    __syncthreads();

    float log_sum = logf(ssum);

    // step 3: compute output = x - max - log_sum
    for (int s = tid; s < seq; s += nthreads)
    {
        float val = input[(b * seq + s) * embedding_dim + e];
        output[(b * seq + s) * embedding_dim + e] = val - max_val - log_sum;
    }
}

// Launcher for embedding_forward
// input: int64_t* on device
// weight: float* on device
// output: float* on device (allocated by binding)
// batch, seq, embedding_dim, num_embeddings (not needed if weight is valid)
// stream: current CUDA stream
extern "C" void embedding_launcher(
    const int64_t* input,
    const float* weight,
    float* output,
    int batch,
    int seq,
    int embedding_dim,
    int num_embeddings,
    cudaStream_t stream)
{
    int total_indices = batch * seq;
    // choose grid size: enough blocks to cover total_indices with 256 threads, but cap at 1024 blocks?
    // grid-stride loop handles any size.
    int block_size = 256;
    int grid_size = (total_indices + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535; // max grid x dimension
    embedding_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weight, output, total_indices, embedding_dim);
}

// Launcher for log_softmax_forward
extern "C" void log_softmax_launcher(
    const float* input,
    float* output,
    int batch,
    int seq,
    int embedding_dim,
    int dim,
    cudaStream_t stream)
{
    // only dim=1 is supported in this kernel; dim may be passed for validation but not used otherwise
    // grid: one block per (batch, embedding) column, each block reduces over seq.
    // block_size fixed to 256, must be power of 2 and >= seq? seq=1024, so 256 threads need multiple iterations.
    int block_size = 256;
    int grid_size = batch * embedding_dim;
    log_softmax_dim1_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, batch, seq, embedding_dim);
}
