// custom_ops.cu - CUDA kernels for abs and log_softmax
#include <cuda_runtime.h>
#include <float.h>

// Element-wise absolute value kernel
template <typename T>
__global__ void abs_kernel(const T* __restrict__ input, T* __restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = fabsf(static_cast<float>(input[i]));
    }
}

// Kernel to compute max value across entire tensor
__global__ void max_reduce_kernel(const float* __restrict__ input, float* __restrict__ partial_max, int64_t numel) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + tid;
    float local_max = -FLT_MAX;

    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        local_max = fmaxf(local_max, val);
    }

    smem[tid] = local_max;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_max[blockIdx.x] = smem[0];
    }
}

// Kernel to compute final max from partial maxes
__global__ void final_max_reduce_kernel(const float* __restrict__ partial_max, float* __restrict__ result, int num_blocks) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        local_max = fmaxf(local_max, partial_max[i]);
    }
    smem[tid] = local_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = smem[0];
    }
}

// Kernel to compute sum of exp(x - max) across tensor, given max_val as pointer
__global__ void logsumexp_reduce_kernel(const float* __restrict__ input, float* __restrict__ partial_sum,
                                        const float* __restrict__ max_val, int64_t numel) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int64_t idx = blockIdx.x * blockDim.x + tid;
    float local_sum = 0.0f;
    float max_v = *max_val;

    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        local_sum += __expf(input[i] - max_v);
    }

    smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sum[blockIdx.x] = smem[0];
    }
}

// Kernel to reduce partial sums to final log_sum_exp (sum, then log)
__global__ void final_reduce_kernel(const float* __restrict__ partial_sum, float* __restrict__ result, int num_blocks) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        local_sum += partial_sum[i];
    }
    smem[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[0] = logf(smem[0]);
    }
}

// Kernel to compute final log_softmax output
__global__ void log_softmax_kernel(const float* __restrict__ input, float* __restrict__ output,
                                   float max_val, float log_sum_exp, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = input[i] - max_val - log_sum_exp;
    }
}

extern "C" void abs_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;
    abs_kernel<float><<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void log_softmax_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream) {
    if (numel <= 0) return;

    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;

    // Allocate temporary device memory for partial reductions
    float *d_partial_max, *d_partial_sum, *d_max_result, *d_log_sum_exp;
    cudaMalloc(&d_partial_max, blocks * sizeof(float));
    cudaMalloc(&d_partial_sum, blocks * sizeof(float));
    cudaMalloc(&d_max_result, sizeof(float));
    cudaMalloc(&d_log_sum_exp, sizeof(float));

    // Step 1: compute per-block max
    max_reduce_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(input, d_partial_max, numel);
    // Reduce partial maxes to single max
    final_max_reduce_kernel<<<1, threads, threads * sizeof(float), stream>>>(d_partial_max, d_max_result, blocks);

    // Step 2: compute per-block sum of exp(x - max)
    logsumexp_reduce_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(input, d_partial_sum, d_max_result, numel);
    // Reduce partial sums to final log_sum_exp
    final_reduce_kernel<<<1, threads, threads * sizeof(float), stream>>>(d_partial_sum, d_log_sum_exp, blocks);

    // Step 3: compute output
    log_softmax_kernel<<<blocks, threads, 0, stream>>>(input, output, 0.0f, 0.0f, numel);

    // Free temporary memory after kernels complete (stream ordered)
    cudaFree(d_partial_max);
    cudaFree(d_partial_sum);
    cudaFree(d_max_result);
    cudaFree(d_log_sum_exp);
}
