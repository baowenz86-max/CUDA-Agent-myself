#include <cuda_runtime.h>
#include <cfloat>

// Kernel: softmax along dim=0 for 2D contiguous tensor (rows x cols)
// Each block handles one column, threads iterate over rows in a grid-stride loop.
__global__ void softmax_dim0_kernel(float* __restrict__ out, const float* __restrict__ in, int rows, int cols) {
    int col = blockIdx.x;
    if (col >= cols) return;

    // Find max per column
    float max_val = -FLT_MAX;
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        float v = in[row * cols + col];
        if (v > max_val) max_val = v;
    }
    // Warp/block reduction for max (since blockDim <= 1024)
    __shared__ float smax[1024];
    smax[threadIdx.x] = max_val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float other = smax[threadIdx.x + stride];
            if (other > smax[threadIdx.x]) smax[threadIdx.x] = other;
        }
        __syncthreads();
    }
    max_val = smax[0];

    // Compute exp and sum
    float sum = 0.0f;
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        float e = __expf(in[row * cols + col] - max_val);
        out[row * cols + col] = e;
        sum += e;
    }
    // Reduce sum
    __shared__ float ssum[1024];
    ssum[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            ssum[threadIdx.x] += ssum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    sum = ssum[0];
    float inv_sum = 1.0f / sum;

    // Normalize
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        out[row * cols + col] *= inv_sum;
    }
}

// Launcher for softmax dim=0
extern "C" void custom_softmax_launcher(float* out, const float* in, int rows, int cols, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = cols;
    softmax_dim0_kernel<<<blocks, threads, 0, stream>>>(out, in, rows, cols);
}
