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
    int total_out = N * C * outH * outW;
    int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int out_idx = idx; out_idx < total_out; out_idx += stride) {
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
// -----------------------------------------------------------------------------
__global__ void flatten_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int N, int C, int H, int W,
                               int sN, int sC, int sH, int sW) {
    int total = N * C * H * W;
    int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int linear_idx = idx; linear_idx < total; linear_idx += stride) {
        // Decode linear index into n,c,h,w assuming row-major contiguous order
        int w = linear_idx % W;
        int h = (linear_idx / W) % H;
        int c = (linear_idx / (W * H)) % C;
        int n = linear_idx / (W * H * C);

        int input_offset = n * sN + c * sC + h * sH + w * sW;
        output[linear_idx] = input[input_offset];
    }
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
// Linear kernel (not used if we use cuBLAS, but kept for completeness)
// -----------------------------------------------------------------------------
#define TILE_M 16
#define TILE_N 10  // because N=10
#define TILE_K 32

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
    dim3 blockDim(TILE_N, TILE_M); // (10,16) = 160 threads, not 256
    dim3 gridDim((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    linear_kernel_v2<<<gridDim, blockDim, 0, stream>>>(input, weight, bias, output, M, N, K);
}
