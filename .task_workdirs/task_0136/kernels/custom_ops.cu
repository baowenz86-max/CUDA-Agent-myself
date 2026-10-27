#include <cuda_runtime.h>
#include <cstdint>

// Helper device function for sigmoid
__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// -----------------------------------------------------------------------------
// Sigmoid forward kernel: y[i] = 1/(1+exp(-x[i]))
// -----------------------------------------------------------------------------
__global__ void sigmoid_kernel(float* __restrict__ output,
                               const float* __restrict__ input,
                               int numel) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < numel; i += stride) {
        output[i] = sigmoidf(input[i]);
    }
}

extern "C" void sigmoid_forward_launcher(
    float* output,
    const float* input,
    int numel,
    cudaStream_t stream) {
    const int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    // Cap blocks to avoid too many blocks for small numel, but grid-stride handles large.
    if (blocks > 65535) blocks = 65535;
    sigmoid_kernel<<<blocks, threads, 0, stream>>>(output, input, numel);
}

// -----------------------------------------------------------------------------
// Average Pooling 2D kernel (kernel_size=2, stride=2, padding=0)
// Input: [N, C, H, W], Output: [N, C, H/2, W/2]
// Each thread computes one output element.
// -----------------------------------------------------------------------------
__global__ void avg_pool2d_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int N, int C, int H, int W,
                                  int outH, int outW) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C * outH * outW;
    if (out_idx >= total_out) return;

    // Decode output index
    int ow = out_idx % outW;
    int oh = (out_idx / outW) % outH;
    int c = (out_idx / (outW * outH)) % C;
    int n = out_idx / (outW * outH * C);

    int h_start = oh * 2;
    int w_start = ow * 2;

    float sum = 0.0f;
    for (int kh = 0; kh < 2; ++kh) {
        int h = h_start + kh;
        if (h >= H) continue; // though without padding and exact divisible, this is always valid
        for (int kw = 0; kw < 2; ++kw) {
            int w = w_start + kw;
            if (w >= W) continue;
            sum += input[((n * C + c) * H + h) * W + w];
        }
    }
    // Since kernel=2 and output dims are exactly H/2, W/2, the count is always 4.
    output[out_idx] = sum * 0.25f;
}

extern "C" void avg_pool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    cudaStream_t stream) {
    int total_out = N * C * outH * outW;
    const int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    avg_pool2d_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W, outH, outW);
}

// -----------------------------------------------------------------------------
// Flatten kernel: from input shape [N, C, H, W] (possibly non-contiguous) to
// output shape [N, C*H*W], preserving row-major order of original contiguous layout.
// Since input may be non-contiguous, we must respect its strides.
// We assume input is the output of avg_pool2d, which is contiguous, but for safety
// we handle general strides.
// The launcher will receive the input pointer and strides as int64_t? We'll keep int.
// -----------------------------------------------------------------------------
__global__ void flatten_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int N, int C, int H, int W,
                               int sN, int sC, int sH, int sW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    // Decode linear index into n,c,h,w assuming row-major contiguous order
    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (W * H)) % C;
    int n = idx / (W * H * C);

    int input_offset = n * sN + c * sC + h * sH + w * sW;
    output[idx] = input[input_offset];
}

extern "C" void flatten_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int sN, int sC, int sH, int sW,
    cudaStream_t stream) {
    int total = N * C * H * W;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    flatten_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W, sN, sC, sH, sW);
}

// -----------------------------------------------------------------------------
// Linear kernel: output = input @ weight^T + bias
// dimensions: input [M, K], weight [N, K], bias [N], output [M, N]
// We use a tiled matrix multiplication with shared memory, each block computes a tile of output.
// Since N=10 is small, we can have each block compute one row of output? Or use a simpler kernel.
// For performance, we use one block per output tile, with blockDim = (TILE_N, TILE_M).
// However, N=10, M=512, K=49152. We can set TILE_N=10, TILE_M=16, and compute per block.
// We'll create a generic kernel where gridDim.y = ceil(M/TILE_M), gridDim.x = ceil(N/TILE_N).
// Each block computes TILE_M x TILE_N outputs using shared memory tiles for A and B.
// Since K is large, we loop over K in chunks.
// -----------------------------------------------------------------------------
#define TILE_M 16
#define TILE_N 10  // because N=10
#define TILE_K 32

__global__ void linear_kernel(const float* __restrict__ input,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ output,
                              int M, int N, int K) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load tile of A (input) from global to shared
        if (row < M && (k0 + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = input[row * K + k0 + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile of B (weight^T) from global to shared: weight is [N, K], so B[k, n] = weight[n, k]
        // We need Bs[local_k][n] = weight[n, k0 + local_k]
        if (col < N && (k0 + threadIdx.y) < K) {
            // threadIdx.y is 0..TILE_M-1, but we need to fill Bs dimensions TILE_K x TILE_N.
            // To fully load Bs, each thread should load one element.
            // We have blockDim.x = TILE_N, blockDim.y = TILE_M, total threads = TILE_N*TILE_M >= TILE_K*TILE_N? not necessarily.
            // Simplification: use blockDim.x=TILE_N, blockDim.y=TILE_M, but for loading B we need all threads cooperate.
            // Better: use a separate loading approach with a loop.
            // Since N=10 is small, we can allocate Bs as [TILE_K][TILE_N] and load using a loop over k.
            // We'll redesign: each thread loads multiple elements if needed.
            // For simplicity, we can assume blockDim.x = TILE_N, blockDim.y = TILE_M, and TILE_M*TILE_N = 160 threads.
            // TILE_K*TILE_N = 320 elements, so each thread loads 2 elements for B.
            // We'll implement loops.
        }
    }
    // This kernel needs careful design. To avoid complexity, we can use a simpler non-tiled kernel with one thread per output element,
    // but K is large (49152) causing many loads. However, N=10, M=512, total outputs=5120, each requires 49152 multiplications,
    // total 251M ops. A naive kernel may be slower than torch.compile but may still meet 5% if other ops dominate? Probably not,
    // but we don't have binding yet to test performance. We'll implement a reasonably optimized version.
    // Alternative: use cuBLAS via cublasSgemm, but we are in .cu and can call cublas. We can include cublas_v2.h and use cublas handle.
    // However, the launcher would need a cublas handle, which we can create/destroy? Better to keep pure CUDA.
    // Given the small N=10, we can have each block compute one column (10 outputs) for a tile of rows, using warp-level reduction?
    // We'll implement a tiled kernel with proper loading loops, but it may be overkill. Let's implement a simpler one:
    // Each thread computes one output element by looping over K. Grid = (ceil(N/??), ceil(M/??)) with blockDim multiple.
    // But to use shared memory, we can have blockDim.x = TILE_N, blockDim.y = TILE_M, and load tiles.
    // We'll write a correct kernel with shared memory, but may not be fully optimized. That's acceptable.
    // We'll restart the linear kernel implementation from scratch below.
}

__global__ void linear_kernel_v2(const float* __restrict__ input,
                                 const float* __restrict__ weight,
                                 const float* __restrict__ bias,
                                 float* __restrict__ output,
                                 int M, int N, int K) {
    __shared__ float As[TILE_M][TILE_K];
    __shared__ float Bs[TILE_K][TILE_N];

    int blockRow = blockIdx.y * TILE_M;
    int blockCol = blockIdx.x * TILE_N;

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;

    float sum = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Load A tile: As[threadRow][threadCol] = input[(blockRow+threadRow)*K + k0+threadCol]
        int globalRow = blockRow + threadRow;
        int globalColA = k0 + threadCol;
        if (globalRow < M && globalColA < K) {
            As[threadRow][threadCol] = input[globalRow * K + globalColA];
        } else {
            As[threadRow][threadCol] = 0.0f;
        }

        // Load B tile: Bs[threadRow][threadCol] = weight[(blockCol+threadCol)*K + k0+threadRow]
        // Note: weight is [N, K], so weight[n, k]. We want Bs[k][n] = weight[n, k].
        // Here threadRow indexes k (0..TILE_K-1) and threadCol indexes n (0..TILE_N-1).
        int globalColB = blockCol + threadCol; // n index
        int globalRowB = k0 + threadRow;       // k index
        if (globalColB < N && globalRowB < K) {
            Bs[threadRow][threadCol] = weight[globalColB * K + globalRowB];
        } else {
            Bs[threadRow][threadCol] = 0.0f;
        }

        __syncthreads();

        // Compute partial sum: sum += As[threadRow][k] * Bs[k][threadCol]
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[threadRow][k] * Bs[k][threadCol];
        }

        __syncthreads();
    }

    int globalRowOut = blockRow + threadRow;
    int globalColOut = blockCol + threadCol;
    if (globalRowOut < M && globalColOut < N) {
        float value = sum + bias[globalColOut];
        output[globalRowOut * N + globalColOut] = value;
    }
}

extern "C" void linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int M, int N, int K,
    cudaStream_t stream) {
    dim3 blockDim(TILE_N, TILE_M); // (16,16) = 256 threads
    dim3 gridDim((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    linear_kernel_v2<<<gridDim, blockDim, 0, stream>>>(input, weight, bias, output, M, N, K);
}
