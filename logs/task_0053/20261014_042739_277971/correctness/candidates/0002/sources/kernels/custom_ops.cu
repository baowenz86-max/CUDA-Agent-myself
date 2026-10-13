#include <cuda_runtime.h>
#include <float.h>

// -----------------------------------------------------------------------------
// Softmax / LogSoftmax along dimension 1 (C) for 3D tensor [B, C, D]
// Input may have arbitrary strides (given as stride_b, stride_c, stride_d)
// Output has same shape and is assumed contiguous (stride_c_out = D, etc.) but we'll use same strides as input for simplicity; output allocated as contiguous by binding layer.
// We'll use one block per (b, d) pair, with blockDim.x covering C.
// If C > blockDim.x, we use grid-stride loop within block and shared memory reduction.
// For C up to 1024, use 512 threads per block and two passes.
// We'll implement a generic reduction with shared memory.

__global__ void softmax_kernel(const float* __restrict__ input, float* __restrict__ output,
                               int B, int C, int D,
                               long long stride_b, long long stride_c, long long stride_d,
                               long long out_stride_b, long long out_stride_c, long long out_stride_d) {
    int b = blockIdx.y;
    int d = blockIdx.x;
    if (b >= B || d >= D) return;

    // index of first element along C for this (b,d)
    const long long base = b * stride_b + d * stride_d;
    const long long out_base = b * out_stride_b + d * out_stride_d;

    // Find max along C for numerical stability
    float max_val = -FLT_MAX;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float val = input[base + c * stride_c];
        max_val = fmaxf(max_val, val);
    }
    // Block reduction for max
    __shared__ float smem[512];
    int tid = threadIdx.x;
    smem[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();

    // Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        float val = input[base + c * stride_c];
        sum += __expf(val - max_val);
    }
    // Block reduction for sum
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    sum = smem[0];
    __syncthreads();

    // Write softmax output
    for (int c = tid; c < C; c += blockDim.x) {
        float val = input[base + c * stride_c];
        output[out_base + c * out_stride_c] = __expf(val - max_val) / sum;
    }
}

__global__ void log_softmax_kernel(const float* __restrict__ input, float* __restrict__ output,
                                   int B, int C, int D,
                                   long long stride_b, long long stride_c, long long stride_d,
                                   long long out_stride_b, long long out_stride_c, long long out_stride_d) {
    int b = blockIdx.y;
    int d = blockIdx.x;
    if (b >= B || d >= D) return;

    const long long base = b * stride_b + d * stride_d;
    const long long out_base = b * out_stride_b + d * out_stride_d;

    float max_val = -FLT_MAX;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float val = input[base + c * stride_c];
        max_val = fmaxf(max_val, val);
    }
    __shared__ float smem[512];
    int tid = threadIdx.x;
    smem[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();

    float sum = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        float val = input[base + c * stride_c];
        sum += __expf(val - max_val);
    }
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    sum = smem[0];
    __syncthreads();

    for (int c = tid; c < C; c += blockDim.x) {
        float val = input[base + c * stride_c];
        output[out_base + c * out_stride_c] = (val - max_val) - logf(sum);
    }
}

// -----------------------------------------------------------------------------
// KL Divergence Loss with batchmean reduction
// input_log: log_softmax output [B, C, D], target_prob: softmax target [B, C, D]
// Returns sum(target_prob * (log(target_prob) - input_log)) / B
// We'll do a two-stage reduction: each thread processes multiple elements, then block reduction, then atomicAdd to global.

__global__ void kl_div_loss_kernel(const float* __restrict__ input_log, const float* __restrict__ target_prob,
                                   float* __restrict__ output, int B, int C, int D,
                                   long long stride_b, long long stride_c, long long stride_d) {
    __shared__ float smem[512];
    int tid = threadIdx.x;
    int total = B * C * D;
    float sum = 0.0f;
    // Each thread sums over strided elements
    for (int idx = blockIdx.x * blockDim.x + tid; idx < total; idx += blockDim.x * gridDim.x) {
        int b = idx / (C * D);
        int rem = idx % (C * D);
        int c = rem / D;
        int d = rem % D;
        long long offset = b * stride_b + c * stride_c + d * stride_d;
        float log_p = input_log[offset];
        float p = target_prob[offset];
        sum += p * (logf(p) - log_p);
    }
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(output, smem[0] / (float)B);
    }
}

// -----------------------------------------------------------------------------
// Cross Entropy Loss with mean reduction
// input: [B, C, D] float32, target: [B, D] int64 class indices
// Returns mean over all B*D positions of -log(softmax(input)[target])
// We compute per-position loss by first finding max along C, then sum exp, then compute loss.
// Use one block per (b, d) to avoid expensive global reductions; block reduces sum of losses and atomicAdd to global.

__global__ void cross_entropy_loss_kernel(const float* __restrict__ input, const int64_t* __restrict__ target,
                                          float* __restrict__ output, int B, int C, int D,
                                          long long stride_b, long long stride_c, long long stride_d,
                                          long long target_stride_b, long long target_stride_d) {
    int b = blockIdx.y;
    int d = blockIdx.x;
    if (b >= B || d >= D) return;

    const long long base = b * stride_b + d * stride_d;
    int t_idx = b * target_stride_b + d * target_stride_d;
    int target_class = (int)target[t_idx];
    if (target_class < 0 || target_class >= C) {
        // Invalid target, but PyTorch would error; we'll just skip contribution (should not happen)
        return;
    }

    // Find max along C for numerical stability
    float max_val = -FLT_MAX;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        max_val = fmaxf(max_val, input[base + c * stride_c]);
    }
    __shared__ float smem[512];
    int tid = threadIdx.x;
    smem[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        }
        __syncthreads();
    }
    max_val = smem[0];
    __syncthreads();

    // Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        sum += __expf(input[base + c * stride_c] - max_val);
    }
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    sum = smem[0];
    __syncthreads();

    // Compute loss for this position: -log(softmax[target_class])
    float target_logit = input[base + target_class * stride_c];
    float loss = -(target_logit - max_val) + logf(sum);

    // Reduce losses across block (all threads have same loss, but we can just have thread 0 atomicAdd)
    if (tid == 0) {
        atomicAdd(output, loss / (float)(B * D));
    }
}

// -----------------------------------------------------------------------------
// Simple scalar add: output = a + b
__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        output[0] = a[0] + b[0];
    }
}

// Simple scalar remainder: output = fmod(input, divisor) (torch.remainder for float uses fmod? Actually remainder as in C fmod for positive divisor; torch.remainder returns remainder with sign of divisor, so for positive divisor it's fmod)
__global__ void remainder_kernel(const float* __restrict__ input, float divisor, float* __restrict__ output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        output[0] = fmodf(input[0], divisor);
    }
}

// -----------------------------------------------------------------------------
// Launchers (extern "C")

extern "C" void log_softmax_forward_launcher(const float* input, float* output,
                                             int B, int C, int D,
                                             long long stride_b, long long stride_c, long long stride_d,
                                             long long out_stride_b, long long out_stride_c, long long out_stride_d,
                                             cudaStream_t stream) {
    dim3 grid(D, B); // (x, y) = (d, b)
    int block = 256; // blockDim.x
    log_softmax_kernel<<<grid, block, 0, stream>>>(input, output, B, C, D,
                                                   stride_b, stride_c, stride_d,
                                                   out_stride_b, out_stride_c, out_stride_d);
}

extern "C" void softmax_forward_launcher(const float* input, float* output,
                                         int B, int C, int D,
                                         long long stride_b, long long stride_c, long long stride_d,
                                         long long out_stride_b, long long out_stride_c, long long out_stride_d,
                                         cudaStream_t stream) {
    dim3 grid(D, B);
    int block = 256;
    softmax_kernel<<<grid, block, 0, stream>>>(input, output, B, C, D,
                                               stride_b, stride_c, stride_d,
                                               out_stride_b, out_stride_c, out_stride_d);
}

extern "C" void kl_div_loss_forward_launcher(const float* input_log, const float* target_prob,
                                             float* output, int B, int C, int D,
                                             long long stride_b, long long stride_c, long long stride_d,
                                             cudaStream_t stream) {
    // Use grid-stride loop with enough blocks
    int total = B * C * D;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535; // cap
    kl_div_loss_kernel<<<grid, block, 0, stream>>>(input_log, target_prob, output, B, C, D,
                                                   stride_b, stride_c, stride_d);
}

extern "C" void cross_entropy_loss_forward_launcher(const float* input, const int64_t* target,
                                                    float* output, int B, int C, int D,
                                                    long long stride_b, long long stride_c, long long stride_d,
                                                    long long target_stride_b, long long target_stride_d,
                                                    cudaStream_t stream) {
    dim3 grid(D, B);
    int block = 256;
    cross_entropy_loss_kernel<<<grid, block, 0, stream>>>(input, target, output, B, C, D,
                                                          stride_b, stride_c, stride_d,
                                                          target_stride_b, target_stride_d);
}

extern "C" void add_forward_launcher(const float* a, const float* b, float* output, cudaStream_t stream) {
    add_kernel<<<1, 1, 0, stream>>>(a, b, output);
}

extern "C" void remainder_forward_launcher(const float* input, float divisor, float* output, cudaStream_t stream) {
    remainder_kernel<<<1, 1, 0, stream>>>(input, divisor, output);
}
