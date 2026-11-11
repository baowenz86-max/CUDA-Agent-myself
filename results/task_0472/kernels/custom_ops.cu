#include <cuda_runtime.h>
#include <float.h>

// Softmax over dimension 1 (H) for a 3D tensor of shape [B, H, W].
// Each thread block computes one (b,w) column of length H.
// We use online softmax to avoid a separate max reduction pass.
// Block size chosen as 256 threads; each thread processes multiple elements via grid-stride loop within the block.
__global__ void softmax_dim1_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int B, int H, int W) {
    int b = blockIdx.x;
    int w = blockIdx.y;
    if (b >= B || w >= W) return;

    const float* in_col = input + b * H * W + w;
    float* out_col = output + b * H * W + w;

    // Online softmax: maintain running max and sum of exponentials.
    float max_val = -FLT_MAX;
    float sum_val = 0.0f;

    int total_threads = blockDim.x;
    int tid = threadIdx.x;
    for (int h = tid; h < H; h += total_threads) {
        float val = in_col[h * W];
        if (val > max_val) {
            // Adjust sum for new max
            sum_val = sum_val * expf(max_val - val) + 1.0f;
            max_val = val;
        } else {
            sum_val += expf(val - max_val);
        }
    }

    // Reduce max and sum across threads in block.
    __shared__ float s_max[256];
    __shared__ float s_sum[256];
    if (tid < total_threads) {
        s_max[tid] = max_val;
        s_sum[tid] = sum_val;
    }
    __syncthreads();

    // Block reduction for max
    for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max[tid + stride] > s_max[tid]) {
                // Update max and adjust sum (note: this is approximate due to online method, but works if we recompute sum later?)
                // Better to recompute sum after global max is known.
                // We'll store max first, then recompute sum in second pass.
                s_max[tid] = s_max[tid + stride];
                // We cannot adjust s_sum easily without exp values; we'll recompute sum later.
            }
        }
        __syncthreads();
    }

    // Actually, to keep correctness and simplicity, we do two-pass approach:
    // 1. Find max within block shared memory.
    // 2. Compute sum of exp(x - max).
    // This is more reliable than online reduction for wide blocks.
    // We'll rewrite below with explicit shared reductions.
}

// Better implementation: two-pass stable softmax with block reduction.
__global__ void softmax_dim1_kernel_v2(const float* __restrict__ input,
                                      float* __restrict__ output,
                                      int B, int H, int W) {
    int b = blockIdx.x;
    int w = blockIdx.y;
    if (b >= B || w >= W) return;

    const float* in_col = input + b * H * W + w;
    float* out_col = output + b * H * W + w;

    extern __shared__ float sdata[]; // size 2*blockDim.x: first max, then sum?
    float* s_max = sdata;
    float* s_sum = sdata + blockDim.x;

    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    // Step 1: compute local max per thread over assigned elements.
    float local_max = -FLT_MAX;
    for (int h = tid; h < H; h += total_threads) {
        float val = in_col[h * W];
        if (val > local_max) local_max = val;
    }
    s_max[tid] = local_max;
    __syncthreads();

    // Block reduction for max
    for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = s_max[tid + stride];
            if (other > s_max[tid]) s_max[tid] = other;
        }
        __syncthreads();
    }
    float block_max = s_max[0];
    __syncthreads(); // ensure all threads read block_max

    // Step 2: compute local sum of exp(x - block_max)
    float local_sum = 0.0f;
    for (int h = tid; h < H; h += total_threads) {
        float val = in_col[h * W];
        local_sum += expf(val - block_max);
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    // Block reduction for sum
    for (int stride = total_threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    float block_sum = s_sum[0];

    // Step 3: write output = exp(x - max) / sum
    for (int h = tid; h < H; h += total_threads) {
        float val = in_col[h * W];
        out_col[h * W] = expf(val - block_max) / block_sum;
    }
}

// ReLU kernel: output = max(input, 0)
__global__ void relu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = input[idx];
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

// Launcher for softmax over dimension 1.
// Input 'input' is assumed to be a contiguous float32 tensor of shape [B, H, W] on CUDA.
// Output 'output' must be allocated with same shape.
extern "C" void softmax_dim1_forward_launcher(const float* input,
                                              float* output,
                                              int B, int H, int W,
                                              cudaStream_t stream) {
    if (B <= 0 || H <= 0 || W <= 0) return;

    // Choose block size: 256 threads (maximum for our shared memory usage = 2*256*4 = 2KB, fine).
    int threads = 256;
    dim3 grid(B, W); // one block per (b,w)
    size_t shmem_size = 2 * threads * sizeof(float);

    softmax_dim1_kernel_v2<<<grid, threads, shmem_size, stream>>>(
        input, output, B, H, W);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a pure C launcher, we cannot throw; but binding will check after call?
        // We can at least print? Better to let binding check.
        // But we can maybe call cudaGetLastError and ignore; binding can also check.
    }
}

// Launcher for ReLU element-wise.
extern "C" void relu_forward_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      cudaStream_t stream) {
    if (numel <= 0) return;

    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore or handle
    }
}
