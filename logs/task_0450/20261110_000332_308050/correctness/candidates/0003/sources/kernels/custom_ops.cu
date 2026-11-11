#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// Global cuBLAS handle, initialized once
static cublasHandle_t global_cublas_handle = nullptr;

// Ensure handle is initialized
static void ensure_cublas_handle() {
    if (global_cublas_handle == nullptr) {
        cublasCreate(&global_cublas_handle);
    }
}

// ======================= Embedding Lookup =======================
__global__ void embedding_lookup_kernel(const int64_t* __restrict__ input,
                                        const float* __restrict__ weight,
                                        float* __restrict__ output,
                                        int num_embeddings,
                                        int embedding_dim,
                                        long total_elements) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (long i = idx; i < total_elements; i += stride) {
        long in_idx = i / embedding_dim;
        long k = i % embedding_dim;
        long row = input[in_idx];
        output[i] = weight[row * embedding_dim + k];
    }
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
        input, weight, output, B * L, E, num_elements);
}

// ======================= Linear Forward =======================
// Uses cuBLAS Sgemm: output = input (M x K) * weight^T (K x N) + bias
// input is [M, K], weight is [N, K] (row-major), output [M, N]
// bias is optional pointer (may be nullptr)

__global__ void add_bias_kernel(float* output,
                                const float* bias,
                                int M,
                                int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    for (int j = 0; j < N; ++j) {
        output[(long)idx * N + j] += bias[j];
    }
}

extern "C" void linear_launcher(const float* input,
                                const float* weight,
                                const float* bias,
                                float* output,
                                int M,
                                int K,
                                int N,
                                cudaStream_t stream) {
    ensure_cublas_handle();
    cublasHandle_t handle = global_cublas_handle;
    cublasSetStream(handle, stream);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t status = cublasSgemm(handle,
                                        CUBLAS_OP_N, CUBLAS_OP_T,
                                        M, N, K,
                                        &alpha,
                                        input, M,
                                        weight, K,
                                        &beta,
                                        output, M);
    if (status != CUBLAS_STATUS_SUCCESS) {
        // Since we cannot throw here, just ignore; but we can check after kernel
        // For debugging, we could use printf, but it's not recommended in production.
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Ignore for now
    }
    if (bias != nullptr) {
        int block = 256;
        int grid = (M + block - 1) / block;
        add_bias_kernel<<<grid, block, 0, stream>>>(output, bias, M, N);
    }
}

// ======================= Cross Entropy Loss =======================
// Input: [B, L, C] contiguous, target: [B, L] int64
// Output: scalar float loss (mean over all B*L positions)
// Each block processes one batch sample (one row of B), positions L over threads+loop
// Uses shared memory to compute partial sums per warp/block, then atomicAdd to global.

#define CROSS_ENTROPY_BLOCK_SIZE 256

__global__ void cross_entropy_loss_kernel(const float* __restrict__ input,
                                          const int64_t* __restrict__ target,
                                          float* __restrict__ loss,
                                          int B,
                                          int L,
                                          int C) {
    // This kernel is not used; see final version below.
}

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

    int threads = CROSS_ENTROPY_BLOCK_SIZE;
    int shared_mem_size = (C + 64) * sizeof(float);

    cross_entropy_loss_kernel_final<<<B, threads, shared_mem_size, stream>>>(
        input, target, loss, B, L, C);

    divide_by_count_kernel<<<1, 1, 0, stream>>>(loss, (long)B * L);
}
