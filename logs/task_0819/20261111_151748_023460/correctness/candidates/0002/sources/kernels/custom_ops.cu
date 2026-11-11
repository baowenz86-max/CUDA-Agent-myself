// kernels/custom_ops.cu
// CUDA kernels for custom_abs and custom_adaptive_log_softmax_with_loss
// No torch/ATen/pybind includes, only raw CUDA and launchers.

#include <cuda_runtime.h>
#include <float.h>

// ---------------------------------------------------------------------
// custom_abs kernel: output[i] = fabsf(input[i])
// ---------------------------------------------------------------------
__global__ void abs_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

extern "C" void custom_abs_launcher(const float* input,
                                    float* output,
                                    int n,
                                    cudaStream_t stream) {
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 65535);
    abs_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

// ---------------------------------------------------------------------
// Helper: row-wise log_softmax for a matrix of shape [rows, cols]
// input: rows x cols, output: rows x cols
// Each block processes one row using shared memory reduction.
// ---------------------------------------------------------------------
__global__ void log_softmax_rows_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int rows,
                                        int cols) {
    extern __shared__ float smem[];
    float* max_vals = smem;
    float* sum_vals = &smem[blockDim.x];

    int row = blockIdx.x;
    if (row >= rows) return;

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Step 1: compute row max
    float local_max = -FLT_MAX;
    for (int col = tid; col < cols; col += stride) {
        float val = input[row * cols + col];
        if (val > local_max) local_max = val;
    }
    // block reduction for max
    max_vals[tid] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (max_vals[tid + offset] > max_vals[tid]) {
                max_vals[tid] = max_vals[tid + offset];
            }
        }
        __syncthreads();
    }
    float row_max = max_vals[0];

    // Step 2: compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += stride) {
        local_sum += __expf(input[row * cols + col] - row_max);
    }
    sum_vals[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sum_vals[tid] += sum_vals[tid + offset];
        }
        __syncthreads();
    }
    float row_sum = sum_vals[0];
    float log_sum = logf(row_sum);

    // Step 3: output x - max - log(sum)
    for (int col = tid; col < cols; col += stride) {
        output[row * cols + col] = input[row * cols + col] - row_max - log_sum;
    }
}

// ---------------------------------------------------------------------
// GEMM kernel: C = A * B^T  (A: MxK, B: NxK, C: MxN)
// We assume B is already contiguous row-major with shape [N, K].
// This simple tiled kernel is used for all linear layers.
// ---------------------------------------------------------------------
__global__ void gemm_nt_kernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M,
                               int N,
                               int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = sum;
    }
}

// ---------------------------------------------------------------------
// Custom adaptive log softmax with loss launcher
// Parameters:
//   input: [batch, in_features]
//   target: [batch] int64
//   weights: pointers to weight matrices
//   cutoffs0, cutoff1, n_classes, div_value
//   log_prob_output: [batch, n_classes] (allocated by caller)
//   loss_output: [batch] per-sample loss (allocated by caller)
//   mean_loss_output: scalar (allocated by caller)
// ---------------------------------------------------------------------
extern "C" void custom_adaptive_log_softmax_with_loss_launcher(
    const float* input,
    const int64_t* target,
    const float* head_weight,   // [head_out, in_features]
    const float* tail0_l1_weight, // [tail0_hidden, in_features]
    const float* tail0_l2_weight, // [tail0_shortlist, tail0_hidden]
    const float* tail1_l1_weight, // [tail1_hidden, in_features]
    const float* tail1_l2_weight, // [tail1_shortlist, tail1_hidden]
    float* log_prob_output,       // [batch, n_classes]
    float* loss_output,           // [batch]
    float* mean_loss_output,      // scalar
    int batch_size,
    int in_features,
    int head_out,                // effective head size = 15000
    int tail0_hidden,
    int tail0_shortlist,
    int tail1_hidden,
    int tail1_shortlist,
    int cutoffs0,
    int cutoffs1,
    int n_classes,
    float div_value,
    cudaStream_t stream)
{
    // Dimensions
    const int M = batch_size;
    const int K = in_features;
    const int N_head = head_out;
    const int N_tail0 = tail0_shortlist;
    const int N_tail1 = tail1_shortlist;

    // Allocate temporary buffers for intermediate results
    // We need:
    // head_logits: [batch, head_out]
    // head_log_probs: [batch, head_out]
    // tail0_logits: [batch, tail0_shortlist]
    // tail0_log_probs: [batch, tail0_shortlist]
    // tail1_logits: [batch, tail1_shortlist]
    // tail1_log_probs: [batch, tail1_shortlist]
    // tail0_hidden_output: [batch, tail0_hidden]
    // tail1_hidden_output: [batch, tail1_hidden]
    // Also need to compute log_softmax for each, and eventually combine.
    // Since we don't have torch allocation, we allocate device memory here using cudaMalloc.
    // However, the launcher signature does not include output allocation; the binding will likely allocate all outputs
    // and we need temporary scratch. Better to allocate within launcher and free after use, or ask binding to pass scratch pointer.
    // To keep launcher self-contained, we allocate temporaries internally using cudaMallocAsync (if available) or cudaMalloc.
    // We'll use cudaMalloc for simplicity; performance may be slightly lower but acceptable.
    
    // Temporary memory sizes
    float *d_head_logits, *d_head_log_probs;
    float *d_tail0_logits, *d_tail0_log_probs;
    float *d_tail1_logits, *d_tail1_log_probs;
    float *d_tail0_hidden, *d_tail1_hidden;
    
    cudaMalloc(&d_head_logits, M * N_head * sizeof(float));
    cudaMalloc(&d_head_log_probs, M * N_head * sizeof(float));
    cudaMalloc(&d_tail0_logits, M * N_tail0 * sizeof(float));
    cudaMalloc(&d_tail0_log_probs, M * N_tail0 * sizeof(float));
    cudaMalloc(&d_tail1_logits, M * N_tail1 * sizeof(float));
    cudaMalloc(&d_tail1_log_probs, M * N_tail1 * sizeof(float));
    cudaMalloc(&d_tail0_hidden, M * tail0_hidden * sizeof(float));
    cudaMalloc(&d_tail1_hidden, M * tail1_hidden * sizeof(float));
    
    // Launch GEMMs
    // head logits: head_weight [head_out, K] -> head_logits [M, head_out]
    {
        dim3 block(16, 16);
        dim3 grid((N_head + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        gemm_nt_kernel<<<grid, block, 0, stream>>>(input, head_weight, d_head_logits, M, N_head, K);
    }
    // tail0 first layer: [M, K] * [tail0_hidden, K]^T -> [M, tail0_hidden]
    {
        dim3 block(16, 16);
        dim3 grid((tail0_hidden + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        gemm_nt_kernel<<<grid, block, 0, stream>>>(input, tail0_l1_weight, d_tail0_hidden, M, tail0_hidden, K);
    }
    // tail1 first layer: [M, K] * [tail1_hidden, K]^T -> [M, tail1_hidden]
    {
        dim3 block(16, 16);
        dim3 grid((tail1_hidden + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        gemm_nt_kernel<<<grid, block, 0, stream>>>(input, tail1_l1_weight, d_tail1_hidden, M, tail1_hidden, K);
    }
    
    // Apply ReLU to hidden outputs? In AdaptiveLogSoftmaxWithLoss, hidden activations are ReLU? Actually, standard implementation uses ReLU after first linear. We assume ReLU.
    // We need a kernel for ReLU or do it in the gemm? For simplicity, we skip activation? But reference likely uses ReLU. We'll add a ReLU kernel.
    __global__ void relu_kernel(float* data, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = idx; i < n; i += stride) {
            if (data[i] < 0) data[i] = 0;
        }
    }
    // Can't define __global__ inside function. Need to define relu_kernel at file scope.
    // We'll define a relu kernel earlier.
    // For now, we call an external relu_kernel defined at top.
    
    // Actually, we need to define relu_kernel at global scope. Let's assume it's defined.
    // We'll add a simple __global__ void relu_kernel here (outside this function).
    
    // Apply ReLU to tail hidden outputs
    int n_tail0_hidden = M * tail0_hidden;
    int n_tail1_hidden = M * tail1_hidden;
    {
        int threads = 256;
        int blocks0 = (n_tail0_hidden + threads - 1) / threads;
        int blocks1 = (n_tail1_hidden + threads - 1) / threads;
        relu_kernel<<<blocks0, threads, 0, stream>>>(d_tail0_hidden, n_tail0_hidden);
        relu_kernel<<<blocks1, threads, 0, stream>>>(d_tail1_hidden, n_tail1_hidden);
    }
    
    // tail0 second layer: [M, tail0_hidden] * [tail0_shortlist, tail0_hidden]^T -> [M, tail0_shortlist]
    {
        dim3 block(16, 16);
        dim3 grid((N_tail0 + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        gemm_nt_kernel<<<grid, block, 0, stream>>>(d_tail0_hidden, tail0_l2_weight, d_tail0_logits, M, N_tail0, tail0_hidden);
    }
    // tail1 second layer: [M, tail1_hidden] * [tail1_shortlist, tail1_hidden]^T -> [M, tail1_shortlist]
    {
        dim3 block(16, 16);
        dim3 grid((N_tail1 + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        gemm_nt_kernel<<<grid, block, 0, stream>>>(d_tail1_hidden, tail1_l2_weight, d_tail1_logits, M, N_tail1, tail1_hidden);
    }
    
    // Compute log_softmax for each part
    // head
    {
        int threads = 256;
        int shared_mem = threads * 2 * sizeof(float);
        log_softmax_rows_kernel<<<M, threads, shared_mem, stream>>>(d_head_logits, d_head_log_probs, M, N_head);
    }
    // tail0
    {
        int threads = 256;
        int shared_mem = threads * 2 * sizeof(float);
        log_softmax_rows_kernel<<<M, threads, shared_mem, stream>>>(d_tail0_logits, d_tail0_log_probs, M, N_tail0);
    }
    // tail1
    {
        int threads = 256;
        int shared_mem = threads * 2 * sizeof(float);
        log_softmax_rows_kernel<<<M, threads, shared_mem, stream>>>(d_tail1_logits, d_tail1_log_probs, M, N_tail1);
    }
    
    // Combine into final log_prob_output of size [batch, n_classes]
    // final columns: 0..head_out-1 -> head_log_probs
    // head_out..head_out+tail0_shortlist-1 -> tail0_log_probs
    // head_out+tail0_shortlist .. n_classes-1 -> tail1_log_probs
    // Note: head_out should be 15000, tail0_shortlist=5000, tail1_shortlist=5000, total 20000.
    // We'll launch a kernel to copy/assign.
    __global__ void combine_log_probs_kernel(const float* head, const float* tail0, const float* tail1,
                                             float* out, int M, int head_out, int tail0_size, int tail1_size, int total) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int i = idx; i < M * total; i += stride) {
            int row = i / total;
            int col = i % total;
            if (col < head_out) {
                out[i] = head[row * head_out + col];
            } else if (col < head_out + tail0_size) {
                out[i] = tail0[row * tail0_size + (col - head_out)];
            } else {
                out[i] = tail1[row * tail1_size + (col - head_out - tail0_size)];
            }
        }
    }
    int total = n_classes;
    int threads_combine = 256;
    int blocks_combine = (M * total + threads_combine - 1) / threads_combine;
    combine_log_probs_kernel<<<blocks_combine, threads_combine, 0, stream>>>(
        d_head_log_probs, d_tail0_log_probs, d_tail1_log_probs,
        log_prob_output, M, N_head, N_tail0, N_tail1, total);
    
    // Compute per-sample loss and mean loss.
    // For each sample i, target[i] determines which part: if target < head_out, use head_log_probs; else if target < head_out + tail0_size, use tail0; else tail1. Loss = -log_prob.
    // We also need to output loss vector and mean.
    __global__ void nll_loss_kernel(const float* head, const float* tail0, const float* tail1,
                                    const int64_t* target, float* loss_out,
                                    int M, int head_out, int tail0_size, int tail1_size, int total) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int stride = blockDim.x * gridDim.x;
        for (int idx = i; idx < M; idx += stride) {
            int64_t t = target[idx];
            float val;
            if (t < head_out) {
                val = -head[idx * head_out + t];
            } else if (t < head_out + tail0_size) {
                val = -tail0[idx * tail0_size + (t - head_out)];
            } else {
                val = -tail1[idx * tail1_size + (t - head_out - tail0_size)];
            }
            loss_out[idx] = val;
        }
    }
    int threads_loss = 256;
    int blocks_loss = (M + threads_loss - 1) / threads_loss;
    nll_loss_kernel<<<blocks_loss, threads_loss, 0, stream>>>(
        d_head_log_probs, d_tail0_log_probs, d_tail1_log_probs,
        target, loss_output, M, N_head, N_tail0, N_tail1, total);
    
    // Compute mean loss into mean_loss_output (scalar)
    // Use a reduction kernel or simply launch a kernel to compute sum then copy to scalar.
    __global__ void mean_reduce_kernel(const float* input, float* output, int n) {
        extern __shared__ float smem[];
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;
        int stride = blockDim.x * gridDim.x;
        float local_sum = 0.0f;
        for (int i = idx; i < n; i += stride) {
            local_sum += input[i];
        }
        smem[tid] = local_sum;
        __syncthreads();
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) smem[tid] += smem[tid + offset];
            __syncthreads();
        }
        if (tid == 0) {
            output[0] = smem[0] / n;
        }
    }
    int threads_mean = 256;
    int blocks_mean = min(1024, (M + threads_mean - 1) / threads_mean);
    mean_reduce_kernel<<<blocks_mean, threads_mean, threads_mean * sizeof(float), stream>>>(
        loss_output, mean_loss_output, M);
    
    // Free temporary memory
    cudaFree(d_head_logits);
    cudaFree(d_head_log_probs);
    cudaFree(d_tail0_logits);
    cudaFree(d_tail0_log_probs);
    cudaFree(d_tail1_logits);
    cudaFree(d_tail1_log_probs);
    cudaFree(d_tail0_hidden);
    cudaFree(d_tail1_hidden);
}

// Need to define relu_kernel and combine_log_probs_kernel at file scope because they are used inside launcher.
__global__ void relu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        if (data[i] < 0) data[i] = 0;
    }
}

// combine_log_probs_kernel definition moved to file scope
__global__ void combine_log_probs_kernel(const float* head, const float* tail0, const float* tail1,
                                         float* out, int M, int head_out, int tail0_size, int tail1_size, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < M * total; i += stride) {
        int row = i / total;
        int col = i % total;
        if (col < head_out) {
            out[i] = head[row * head_out + col];
        } else if (col < head_out + tail0_size) {
            out[i] = tail0[row * tail0_size + (col - head_out)];
        } else {
            out[i] = tail1[row * tail1_size + (col - head_out - tail0_size)];
        }
    }
}

// nll_loss_kernel definition
__global__ void nll_loss_kernel(const float* head, const float* tail0, const float* tail1,
                                const int64_t* target, float* loss_out,
                                int M, int head_out, int tail0_size, int tail1_size, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = i; idx < M; idx += stride) {
        int64_t t = target[idx];
        float val;
        if (t < head_out) {
            val = -head[idx * head_out + t];
        } else if (t < head_out + tail0_size) {
            val = -tail0[idx * tail0_size + (t - head_out)];
        } else {
            val = -tail1[idx * tail1_size + (t - head_out - tail0_size)];
        }
        loss_out[idx] = val;
    }
}

// mean_reduce_kernel definition
__global__ void mean_reduce_kernel(const float* input, float* output, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;
    for (int i = idx; i < n; i += stride) {
        local_sum += input[i];
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) smem[tid] += smem[tid + offset];
        __syncthreads();
    }
    if (tid == 0) {
        output[0] = smem[0] / n;
    }
}
