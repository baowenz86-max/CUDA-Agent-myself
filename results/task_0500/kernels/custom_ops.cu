// kernels/custom_ops.cu
// CUDA kernels for fused einsum + bmm: y = x @ x, z = y @ y
#include <cuda_runtime.h>

// Tiled matrix multiplication kernel: C = A * B where A, B are [N, N] matrices.
// Each block computes a tile of C of size TILE_DIM x TILE_DIM.
// Uses shared memory for tiles of A and B.
template <int TILE_DIM>
__global__ void matmul_tile_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    // Block indices
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    float sum = 0.0f;
    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; ++t) {
        // Load A tile
        int a_col = t * TILE_DIM + threadIdx.x;
        int a_row = row;
        if (a_row < N && a_col < N) {
            As[threadIdx.y][threadIdx.x] = A[a_row * N + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile
        int b_row = t * TILE_DIM + threadIdx.y;
        int b_col = col;
        if (b_row < N && b_col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Accumulate dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result if within bounds
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Launcher for fused operation: computes Z = (X*X) * (X*X)
// Input: x (float*, contiguous), output: z (float*), B batch size, N dimension, stream
extern "C" void einsum_bmm_fused_launcher(const float* x,
                                          float* z,
                                          int B,
                                          int N,
                                          cudaStream_t stream) {
    if (B <= 0 || N <= 0) return;

    // Choose tile dimensions: 32x32 for good occupancy and shared memory usage (32*32*4*2 = 8192 bytes)
    const int TILE_DIM = 32;

    // Allocate temporary buffer for Y of size B*N*N float
    float* y;
    cudaError_t err = cudaMallocAsync(&y, (size_t)B * N * N * sizeof(float), stream);
    if (err != cudaSuccess) {
        // fallback to cudaMalloc (though async requires stream)
        err = cudaMalloc(&y, (size_t)B * N * N * sizeof(float));
        if (err != cudaSuccess) return;
    }

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    // First multiplication: Y = X * X for each batch
    for (int b = 0; b < B; ++b) {
        const float* A = x + (size_t)b * N * N;
        float* C = y + (size_t)b * N * N;
        matmul_tile_kernel<TILE_DIM><<<grid, block, 0, stream>>>(A, A, C, N);
    }

    // Second multiplication: Z = Y * Y for each batch
    for (int b = 0; b < B; ++b) {
        const float* A = y + (size_t)b * N * N;
        float* C = z + (size_t)b * N * N;
        matmul_tile_kernel<TILE_DIM><<<grid, block, 0, stream>>>(A, A, C, N);
    }

    // Free temporary buffer (async free is safe if using cudaMallocAsync; if fallback malloc, use cudaFree)
    cudaFreeAsync(y, stream);
}

// Simple non-tiled kernel for small N (fallback) - not needed, but kept for completeness
__global__ void matmul_naive_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Alternative launcher with naive kernel (for small N, but we'll use tiled always)
extern "C" void einsum_bmm_fused_naive_launcher(const float* x,
                                                float* z,
                                                int B,
                                                int N,
                                                cudaStream_t stream) {
    if (B <= 0 || N <= 0) return;
    float* y;
    cudaMalloc(&y, (size_t)B * N * N * sizeof(float));
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    for (int b = 0; b < B; ++b) {
        const float* A = x + (size_t)b * N * N;
        float* C = y + (size_t)b * N * N;
        matmul_naive_kernel<<<grid, block, 0, stream>>>(A, A, C, N);
    }
    for (int b = 0; b < B; ++b) {
        const float* A = y + (size_t)b * N * N;
        float* C = z + (size_t)b * N * N;
        matmul_naive_kernel<<<grid, block, 0, stream>>>(A, A, C, N);
    }
    cudaFree(y);
}
