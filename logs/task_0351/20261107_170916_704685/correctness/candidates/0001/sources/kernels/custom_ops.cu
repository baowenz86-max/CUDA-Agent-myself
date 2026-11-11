#include <cuda_runtime.h>

// Embedding forward launcher
// indices: CPU int64 pointer, shape [batch, seq]
// weight: CUDA float pointer, shape [num_embeddings, embedding_dim]
// output: CUDA float pointer, shape [batch, seq, embedding_dim]
// stream: CUDA stream for kernel launch
extern "C" void embedding_launcher(
    const int64_t* indices,
    const float* weight,
    float* output,
    int batch,
    int seq,
    int embedding_dim,
    cudaStream_t stream)
{
    int total_indices = batch * seq;
    if (total_indices <= 0) return;

    int total_output_elements = total_indices * embedding_dim;
    int threads = 256;
    int blocks = (total_output_elements + threads - 1) / threads;

    // Use grid-stride loop
    int grid_size = blocks;

    // Kernel launch with grid-stride pattern
    embedding_kernel<<<blocks, threads, 0, stream>>>(
        indices, weight, output, total_indices, embedding_dim, total_output_elements);
}

// Kernel: each thread handles one output element, but we need block-based copy for efficiency.
// Better approach: each thread block handles one or more indices and copies embedding_dim floats.
__global__ void embedding_kernel_naive(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int total_indices,
    int embedding_dim,
    int total_output_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_output_elements; i += stride) {
        int index_pos = i / embedding_dim;
        int dim = i % embedding_dim;
        int64_t index = indices[index_pos];
        output[i] = weight[index * embedding_dim + dim];
    }
}

// Optimized kernel: each thread block processes one or more indices,
// threads within block copy consecutive embedding_dim elements.
__global__ void embedding_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int total_indices,
    int embedding_dim,
    int total_output_elements)
{
    extern __shared__ int64_t s_indices[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Process indices in chunks of block_size
    for (int base = blockIdx.x * block_size; base < total_indices; base += gridDim.x * block_size) {
        int idx = base + tid;
        if (idx < total_indices) {
            s_indices[tid] = indices[idx];
        }
        __syncthreads();

        // Copy embedding_dim elements for each index in shared memory
        // Assume embedding_dim is multiple of block_size for simplicity, but handle general case.
        int num_indices_in_block = min(block_size, total_indices - base);
        for (int i = 0; i < num_indices_in_block; ++i) {
            int64_t emb_idx = s_indices[i];
            const float* src = weight + emb_idx * embedding_dim;
            float* dst = output + (base + i) * embedding_dim;
            // Copy embedding_dim elements using threads in block, grid-stride within block
            for (int j = tid; j < embedding_dim; j += block_size) {
                dst[j] = src[j];
            }
        }
        __syncthreads();
    }
}

// But the above kernel uses dynamic shared memory and might be suboptimal.
// Simpler: each thread handles one output element as in naive, but with vectorized loads?
// embedding_dim = 256, so each index copies 256 floats. A block of 256 threads can handle one index perfectly.
// We'll implement a kernel where each block handles one index (or multiple if blockDim > embedding_dim, not needed).
// block_size = 256, each block handles one index, one thread per element.
// This is coalesced for output, input weight access is also coalesced if consecutive indices have consecutive weights? Not necessarily, but okay.

// Let's define final kernel: each thread handles one output element, blockDim=256, grid covers total_output_elements.
// But that may be many blocks. For total_output_elements = 128*4096*256 = 134,217,728 -> blocks = 524,288 if 256 threads.
// That's fine.

// Use a simple grid-stride loop with vectorized float4? Leave as simple.

__global__ void embedding_kernel_simple(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int total_indices,
    int embedding_dim,
    int total_output_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_output_elements) {
        int index_pos = idx / embedding_dim;
        int dim = idx % embedding_dim;
        int64_t index = indices[index_pos];
        output[idx] = weight[index * embedding_dim + dim];
    }
}

// We'll use the simple kernel for correctness first.

// mul_scalar launcher
extern "C" void mul_scalar_launcher(
    const float* input,
    const float* scalar,
    float* output,
    int n,
    cudaStream_t stream)
{
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mul_scalar_kernel<<<blocks, threads, 0, stream>>>(input, scalar, output, n);
}

__global__ void mul_scalar_kernel(
    const float* __restrict__ input,
    const float* __restrict__ scalar,
    float* __restrict__ output,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * scalar[0];
    }
}
