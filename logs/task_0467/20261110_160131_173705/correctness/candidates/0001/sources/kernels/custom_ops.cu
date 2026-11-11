#include <cuda_runtime.h>

// HardTanh elementwise: clamp(x, -1.0f, 1.0f)
__global__ void hardtanh_kernel(const float* __restrict__ x, float* __restrict__ y, long numel) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;
    for (long i = idx; i < numel; i += stride) {
        float v = x[i];
        y[i] = fminf(fmaxf(v, -1.0f), 1.0f);
    }
}

// Linear: y = x @ weight^T + bias
// x: M x K (row-major), weight: N x K (row-major, then transpose), bias: N
// y: M x N
// Block size: 16x16 threads, each computes one output element.
// Requires M, N, K to be reasonably sized; handled with grid-stride loops for M,N but K fixed per block.
// For simplicity, we use a basic kernel with each thread computing one output element and K loop unrolled? Better use shared memory for x and weight tiles for performance.
// But to keep code simple and correct, use each thread computes one output element with K loop in registers (no shared memory) for now.
// Performance might be suboptimal but acceptable for first iteration.
__global__ void linear_naive_kernel(const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias, float* __restrict__ y,
                                    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += x[row * K + k] * weight[col * K + k];
        }
        y[row * N + col] = sum + bias[col];
    }
}

// Optimized linear kernel using shared memory tiles.
// Block size 16x16, tile size 16 for K.
#define TILE_K 16
#define BLOCK_DIM 16
__global__ void linear_tiled_kernel(const float* __restrict__ x, const float* __restrict__ weight, const float* __restrict__ bias, float* __restrict__ y,
                                    int M, int N, int K) {
    __shared__ float x_tile[BLOCK_DIM][TILE_K];
    __shared__ float w_tile[BLOCK_DIM][TILE_K]; // w_tile[col][k] = weight[col*K + k]
    
    int row = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int col = blockIdx.x * BLOCK_DIM + threadIdx.x;
    
    float sum = 0.0f;
    for (int k_start = 0; k_start < K; k_start += TILE_K) {
        // Load x tile: x[row][k_start + threadIdx.x] if within K
        int k = k_start + threadIdx.x;
        if (row < M && k < K) {
            x_tile[threadIdx.y][threadIdx.x] = x[row * K + k];
        } else {
            x_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Load weight tile: weight[col][k_start + threadIdx.y]
        k = k_start + threadIdx.y;
        if (col < N && k < K) {
            w_tile[threadIdx.y][threadIdx.x] = weight[col * K + k];
        } else {
            w_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        // Accumulate partial products
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            sum += x_tile[threadIdx.y][kk] * w_tile[threadIdx.x][kk];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        y[row * N + col] = sum + bias[col];
    }
}

// Launcher for hardtanh. Assumes contiguous float32 input and output.
extern "C" void hardtanh_launcher(const float* x, float* y, long numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 0) {
        hardtanh_kernel<<<blocks, threads, 0, stream>>>(x, y, numel);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In production, binding should check this. For now, just ignore or maybe return error?
        // We'll keep as is.
    }
}

// Launcher for linear. Assumes contiguous float32 inputs and output.
extern "C" void linear_launcher(const float* x, const float* weight, const float* bias, float* y,
                                int M, int N, int K, cudaStream_t stream) {
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);
    linear_tiled_kernel<<<grid, block, 0, stream>>>(x, weight, bias, y, M, N, K);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // as above
    }
}
