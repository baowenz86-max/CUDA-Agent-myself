// CUDA kernels for custom_embedding and custom_baddbmm
// No PyTorch dependencies. Launchers accept raw pointers and cudaStream_t.

#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------
// Embedding lookup kernel
// indices: int64_t, shape [batch_size, seq_len]
// weight: float, shape [num_embeddings, embedding_dim]
// output: float, shape [batch_size, seq_len, embedding_dim]
// ---------------------------------------------------------------------
__global__ void embedding_lookup_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int seq_len,
    int embedding_dim
) {
    // Each thread computes one output element (row = b, col = e)
    // Total elements = batch_size * seq_len * embedding_dim
    int total = batch_size * seq_len * embedding_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < total; i += stride) {
        int e = i % embedding_dim;
        int tmp = i / embedding_dim;
        int s = tmp % seq_len;
        int b = tmp / seq_len;
        
        int64_t token = indices[b * seq_len + s];
        // Assume valid index, no padding_idx support needed
        output[i] = weight[token * embedding_dim + e];
    }
}

// ---------------------------------------------------------------------
// Batched matrix multiply-add kernel (x + batch1 @ batch2)
// x: float, shape [batch, M, N]   (M=seq_len, N=embedding_dim)
// batch1: float, shape [batch, M, K]  (K=input_dim)
// batch2: float, shape [batch, K, N]
// output: float, shape [batch, M, N]
// ---------------------------------------------------------------------
// Block size: 16x16 threads, each computes one output element.
// Shared memory tiles:
//   sA: 16 x 16 float (batch1 tile)
//   sB: 16 x 16 float (batch2 tile)
#define TILE_DIM 16
#define BLOCK_ROWS 16
#define BLOCK_COLS 16

__global__ void baddbmm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ batch1,
    const float* __restrict__ batch2,
    float* __restrict__ output,
    int M,      // seq_len
    int N,      // embedding_dim
    int K       // input_dim
) {
    __shared__ float sA[BLOCK_ROWS][TILE_DIM];
    __shared__ float sB[TILE_DIM][BLOCK_COLS];

    int batch = blockIdx.z;
    int block_row = blockIdx.y * BLOCK_ROWS;
    int block_col = blockIdx.x * BLOCK_COLS;

    int row = block_row + threadIdx.y;
    int col = block_col + threadIdx.x;

    float acc = 0.0f;
    if (row < M && col < N) {
        acc = x[batch * M * N + row * N + col];
    }

    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < num_tiles; ++t) {
        // Load tile from batch1: shape M x K
        int k_start = t * TILE_DIM;
        int k = k_start + threadIdx.x;
        if (row < M && k < K) {
            sA[threadIdx.y][threadIdx.x] = batch1[batch * M * K + row * K + k];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from batch2: shape K x N (transposed in shared memory)
        int k_row = k_start + threadIdx.y;
        if (k_row < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = batch2[batch * K * N + k_row * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        if (row < M && col < N) {
            #pragma unroll
            for (int kk = 0; kk < TILE_DIM; ++kk) {
                acc += sA[threadIdx.y][kk] * sB[kk][threadIdx.x];
            }
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        output[batch * M * N + row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------
// Launcher functions (extern "C")
// ---------------------------------------------------------------------

// custom_embedding launcher:
//   indices_cpu: int64_t pointer on CPU, shape [batch_size * seq_len]
//   weight_cpu: float pointer on CPU, shape [num_embeddings * embedding_dim]
//   output_gpu: float pointer on GPU, shape [batch_size * seq_len * embedding_dim] (preallocated)
//   batch_size, seq_len, embedding_dim, num_embeddings: int scalars
//   stream: cudaStream_t
extern "C" void custom_embedding_launcher(
    const int64_t* indices_cpu,
    const float* weight_cpu,
    float* output_gpu,
    int batch_size,
    int seq_len,
    int embedding_dim,
    int num_embeddings,
    cudaStream_t stream
) {
    size_t indices_bytes = (size_t)batch_size * seq_len * sizeof(int64_t);
    size_t weight_bytes = (size_t)num_embeddings * embedding_dim * sizeof(float);
    
    // Allocate temporary GPU buffers for indices and weight
    int64_t* indices_gpu;
    float* weight_gpu;
    cudaMalloc(&indices_gpu, indices_bytes);
    cudaMalloc(&weight_gpu, weight_bytes);
    
    // Copy inputs from CPU to GPU asynchronously on stream
    cudaMemcpyAsync(indices_gpu, indices_cpu, indices_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(weight_gpu, weight_cpu, weight_bytes, cudaMemcpyHostToDevice, stream);
    
    // Launch kernel
    int total = batch_size * seq_len * embedding_dim;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    // Cap grid size to avoid huge launches, but use grid-stride loop
    if (grid_size > 65535) grid_size = 65535;
    embedding_lookup_kernel<<<grid_size, block_size, 0, stream>>>(
        indices_gpu, weight_gpu, output_gpu,
        batch_size, seq_len, embedding_dim
    );
    
    // Free temporary GPU buffers after kernel completes (on stream)
    cudaFreeAsync(indices_gpu, stream);
    cudaFreeAsync(weight_gpu, stream);
}

// custom_baddbmm launcher:
//   x_gpu: float pointer on GPU, shape [batch * M * N]
//   batch1_cpu: float pointer on CPU, shape [batch * M * K]
//   batch2_cpu: float pointer on CPU, shape [batch * K * N]
//   output_gpu: float pointer on GPU, shape [batch * M * N] (preallocated)
//   batch, M, N, K: int scalars
//   stream: cudaStream_t
extern "C" void custom_baddbmm_launcher(
    const float* x_gpu,
    const float* batch1_cpu,
    const float* batch2_cpu,
    float* output_gpu,
    int batch,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    size_t batch1_bytes = (size_t)batch * M * K * sizeof(float);
    size_t batch2_bytes = (size_t)batch * K * N * sizeof(float);
    
    // Allocate temporary GPU buffers for batch1 and batch2
    float* batch1_gpu;
    float* batch2_gpu;
    cudaMalloc(&batch1_gpu, batch1_bytes);
    cudaMalloc(&batch2_gpu, batch2_bytes);
    
    // Copy batch1 and batch2 from CPU to GPU asynchronously
    cudaMemcpyAsync(batch1_gpu, batch1_cpu, batch1_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(batch2_gpu, batch2_cpu, batch2_bytes, cudaMemcpyHostToDevice, stream);
    
    // Launch kernel
    dim3 block(BLOCK_COLS, BLOCK_ROWS);  // 16x16 threads
    // grid: x-dim for N tiles, y-dim for M tiles, z-dim for batch
    int grid_x = (N + BLOCK_COLS - 1) / BLOCK_COLS;
    int grid_y = (M + BLOCK_ROWS - 1) / BLOCK_ROWS;
    dim3 grid(grid_x, grid_y, batch);
    
    baddbmm_kernel<<<grid, block, 0, stream>>>(
        x_gpu, batch1_gpu, batch2_gpu, output_gpu,
        M, N, K
    );
    
    // Free temporary GPU buffers after kernel completes
    cudaFreeAsync(batch1_gpu, stream);
    cudaFreeAsync(batch2_gpu, stream);
}
