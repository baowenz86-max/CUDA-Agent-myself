#include <cuda_runtime.h>
#include <math.h>

// ======================= Embedding Lookup =======================
// Each thread computes one output element (b, l, e).
__global__ void embedding_lookup_kernel(const int64_t* __restrict__ input,
                                        const float* __restrict__ weight,
                                        float* __restrict__ output,
                                        int B, int L, int E,
                                        long total_elements) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    long e = idx % E;               // embedding dim index
    long pos = idx / E;             // flattened position within B*L
    long in_idx = pos;              // index into input array
    long row = input[in_idx];       // embedding row index
    output[idx] = weight[row * E + e];
}

extern "C" void embedding_lookup_launcher(const int64_t* input,
                                           const float* weight,
                                           float* output,
                                           int B,
                                           int L,
                                           int E,
                                           cudaStream_t stream) {
    long num_elements = (long)B * L * E;
    int block = 256;
    long grid = (num_elements + block - 1) / block;
    if (grid > 2147483647) grid = 2147483647;
    embedding_lookup_kernel<<<(unsigned)grid, block, 0, stream>>>(
        input, weight, output, B, L, E, num_elements);
}

// ======================= Linear Forward (custom GEMM) =======================
// Computes output[m, n] = sum_{k=0}^{K-1} input[m, k] * weight[n, k] + bias[n]
// Input is [M, K] row-major, weight is [N, K] row-major, output is [M, N] row-major.
__global__ void linear_kernel(const float* __restrict__ input,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ output,
                              int M, int K, int N) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)M * N) return;

    int n = idx % N;
    int m = idx / N;

    float sum = 0.0f;
    const float* input_row = input + (long)m * K;
    const float* weight_row = weight + (long)n * K;
    for (int k = 0; k < K; ++k) {
        sum += input_row[k] * weight_row[k];
    }
    if (bias != nullptr) {
        sum += bias[n];
    }
    output[idx] = sum;
}

extern "C" void linear_launcher(const float* input,
                                const float* weight,
                                const float* bias,
                                float* output,
                                int M,
                                int K,
                                int N,
                                cudaStream_t stream) {
    long total = (long)M * N;
    int block = 256;
    long grid = (total + block - 1) / block;
    if (grid > 2147483647) grid = 2147483647;
    linear_kernel<<<(unsigned)grid, block, 0, stream>>>(
        input, weight, bias, output, M, K, N);
}

// ======================= Cross Entropy Loss =======================
// Placeholder implementation; not used for output and will be corrected later.
__global__ void cross_entropy_loss_kernel_final(const float* __restrict__ input,
                                                const int64_t* __restrict__ target,
                                                float* __restrict__ loss,
                                                int B,
                                                int L,
                                                int C) {
    extern __shared__ float smem[];
    int bid = blockIdx.x;
    if (bid >= B) return;

    const float* sample_input = input + (long)bid * L * C;
    const int64_t* sample_target = target + (long)bid * L;

    float* logits = smem;
    float* reduction = smem + C;

    int tid = threadIdx.x;
    int threads = blockDim.x;

    float total_loss_positions = 0.0f;

    for (int pos = 0; pos < L; ++pos) {
        for (int idx = tid; idx < C; idx += threads) {
            logits[idx] = sample_input[(long)pos * C + idx];
        }
        __syncthreads();

        float thread_max = -INFINITY;
        for (int idx = tid; idx < C; idx += threads) {
            float val = logits[idx];
            if (val > thread_max) thread_max = val;
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, thread_max, offset);
            if (other > thread_max) thread_max = other;
        }

        int warp_id = tid / 32;
        int lane_id = tid % 32;
        int num_warps = (threads + 31) / 32;
        __shared__ float warp_max[32];
        if (lane_id == 0) {
            warp_max[warp_id] = thread_max;
        }
        __syncthreads();

        float block_max = -INFINITY;
        if (tid < num_warps) {
            block_max = warp_max[tid];
        }
        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                float other = warp_max[tid + offset];
                if (other > block_max) block_max = other;
                warp_max[tid] = block_max;
            }
            __syncthreads();
        }
        if (tid == 0) {
            block_max = warp_max[0];
            reduction[0] = block_max;
        }
        __syncthreads();
        block_max = reduction[0];

        float thread_sum = 0.0f;
        for (int idx = tid; idx < C; idx += threads) {
            thread_sum += __expf(logits[idx] - block_max);
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }

        if (lane_id == 0) {
            warp_max[warp_id] = thread_sum;
        }
        __syncthreads();

        float block_sum = 0.0f;
        if (tid < num_warps) {
            block_sum = warp_max[tid];
        }
        for (int offset = num_warps / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                block_sum += warp_max[tid + offset];
                warp_max[tid] = block_sum;
            }
            __syncthreads();
        }
        if (tid == 0) {
            reduction[1] = warp_max[0];
        }
        __syncthreads();
        block_sum = reduction[1];

        int target_class = sample_target[pos];
        float target_logit = logits[target_class];
        float log_prob = - (target_logit - block_max) - logf(block_sum);

        if (tid == 0) {
            total_loss_positions += log_prob;
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(loss, total_loss_positions);
    }
}

__global__ void divide_by_count_kernel(float* loss, long count) {
    *loss /= (float)count;
}

extern "C" void cross_entropy_loss_launcher(const float* input,
                                            const int64_t* target,
                                            float* loss,
                                            int B,
                                            int L,
                                            int C,
                                            cudaStream_t stream) {
    cudaMemsetAsync(loss, 0, sizeof(float), stream);

    int threads = 256;
    int shared_mem_size = (C + 64) * sizeof(float);

    cross_entropy_loss_kernel_final<<<B, threads, shared_mem_size, stream>>>(
        input, target, loss, B, L, C);

    divide_by_count_kernel<<<1, 1, 0, stream>>>(loss, (long)B * L);
}
