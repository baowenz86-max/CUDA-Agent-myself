#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdint>

// ------------------------------------------------------------
// Utility: index calculation for 2D row-major tensors
// ------------------------------------------------------------
__device__ __forceinline__ int64_t idx2d(int64_t row, int64_t col, int64_t stride_col) {
    return row * stride_col + col;
}

// ------------------------------------------------------------
// Kernel 1: one_hot_float
// Input: int64_t* x (CPU pointer, but kernel runs on device after copy)
// Output: float* out (device memory), shape [batch_size, num_classes]
// Note: This kernel is launched on GPU; the CPU input must be copied to device
// by the binding code, or we can accept device pointer if binding handles copy.
// Since contract specifies input as CPU tensor, binding must copy to device.
// We provide a launcher that takes device pointer after copy.
// ------------------------------------------------------------
__global__ void one_hot_kernel(const int64_t* __restrict__ x, float* __restrict__ out,
                               int64_t batch_size, int64_t num_classes) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        int64_t cls = x[idx];
        // Initialize entire row to zero (or just set the one-hot value and rely on prior zeroing?)
        // Safer to set all to zero first, but that would require num_classes writes per row.
        // Efficient approach: each thread handles one element? Here we have only batch_size threads,
        // so we need to zero the row then set one element. Better: launch with batch_size * num_classes threads.
        // We'll instead use a 2D grid: each thread handles one element of output.
        // We'll implement a kernel that sets out[row, col] = (x[row] == col) ? 1.0f : 0.0f.
    }
}

// Better one_hot kernel with 2D indexing
__global__ void one_hot_kernel_2d(const int64_t* __restrict__ x, float* __restrict__ out,
                                  int64_t batch_size, int64_t num_classes) {
    int64_t row = blockIdx.y;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < batch_size && col < num_classes) {
        int64_t cls = x[row];
        out[row * num_classes + col] = (col == cls) ? 1.0f : 0.0f;
    }
}

extern "C" void one_hot_float_launcher(const int64_t* x_dev, float* out_dev,
                                       int64_t batch_size, int64_t num_classes,
                                       cudaStream_t stream) {
    // Assume x_dev is device pointer after binding copies CPU tensor to device.
    // Binding is responsible for copying. We just launch kernel.
    dim3 block(256);
    dim3 grid((num_classes + block.x - 1) / block.x, batch_size);
    one_hot_kernel_2d<<<grid, block, 0, stream>>>(x_dev, out_dev, batch_size, num_classes);
}

// ------------------------------------------------------------
// Helper: matrix multiplication using cuBLAS (assumes row-major -> column-major trick)
// We'll use cuBLAS with CUBLAS_OP_T for both A and B to compute C = A^T * B^T etc.
// For our use: we need to compute output = input * weight^T + bias.
// input: [M, K], weight: [N, K], output: [M, N]
// In cuBLAS column-major, this is equivalent to C (N x M) = W (N x K) * X^T (K x M).
// We'll use cublasSgemm with transa=CUBLAS_OP_N, transb=CUBLAS_OP_T, m=N, n=M, k=K,
// alpha=1, A=weight, lda=N, B=input, ldb=M, beta=0, C=output, ldc=N.
// ------------------------------------------------------------

// Helper function to get cuBLAS handle (global)
cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        cublasCreate(&handle);
    }
    return handle;
}

// GEMM launcher: out = input * weight^T
// input: M x K (row-major), weight: N x K (row-major)
// out: M x N (row-major)
void gemm_nt(const float* input, const float* weight, float* out,
             int M, int N, int K, cudaStream_t stream) {
    cublasHandle_t handle = get_cublas_handle();
    cublasSetStream(handle, stream);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Column-major: A is weight (N x K) transposed? We'll use CUBLAS_OP_N for A (weight as N x K),
    // B as input (M x K) but we need B^T. Use CUBLAS_OP_T for B.
    // Let's be consistent: For row-major C = A_row * B_row^T where A_row is M x K, B_row is N x K
    // cuBLAS column-major equivalent: C_col = B_col * A_col^T, where A_col = A_row^T (K x M), B_col = B_row^T (K x N)
    // So we call cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K, &alpha, B_row, N, A_row, M, &beta, C, N)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                N, M, K,
                &alpha,
                weight, N,
                input, M,
                &beta,
                out, N);
}

// GEMM with bias: out = input * weight^T + bias
void gemm_nt_bias(const float* input, const float* weight, const float* bias, float* out,
                  int M, int N, int K, cudaStream_t stream) {
    gemm_nt(input, weight, out, M, N, K, stream);
    // Add bias: each row gets bias added
    // We need a kernel to add bias to each row, or do it in log_softmax kernel later.
    // For now, we'll incorporate bias in subsequent kernels.
}

// ------------------------------------------------------------
// Kernel: head_log_softmax
// Computes log_softmax over the head classes (first head_size columns)
// Input: head_logits [M, head_size], already includes bias.
// Output: head_log_probs [M, head_size]
// Also need to compute max and sum for numerical stability.
// We'll do per-row:
// 1. Find max value in row.
// 2. Compute exp(x - max) and sum.
// 3. Write log_softmax = (x - max) - log(sum).
// We'll use one block per row, with threads cooperating.
// head_size = head_size (302 in example)
// ------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void log_softmax_kernel(const float* __restrict__ logits, float* __restrict__ log_probs,
                                   int rows, int cols) {
    extern __shared__ float shared[];
    float* max_vals = shared;
    float* sum_vals = &shared[BLOCK_SIZE];

    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = logits + row * cols;
    float* out_ptr = log_probs + row * cols;

    // Find max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        max_val = fmaxf(max_val, row_ptr[i]);
    }
    // Block reduction for max
    __shared__ float block_max[BLOCK_SIZE];
    block_max[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_max[threadIdx.x] = fmaxf(block_max[threadIdx.x], block_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = block_max[0];
    __syncthreads();

    // Compute sum of exp(x - max)
    float sum_val = 0.0f;
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        sum_val += __expf(row_ptr[i] - max_val);
    }
    // Block reduction for sum
    __shared__ float block_sum[BLOCK_SIZE];
    block_sum[threadIdx.x] = sum_val;
    __syncthreads();
    for (int s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum_val = block_sum[0];
    __syncthreads();

    float log_sum = logf(sum_val);

    // Write log_softmax
    for (int i = threadIdx.x; i < cols; i += BLOCK_SIZE) {
        out_ptr[i] = (row_ptr[i] - max_val) - log_sum;
    }
}

// ------------------------------------------------------------
// Kernel: compute_tail_log_probs
// For samples assigned to a tail cluster, we need to compute log probabilities
// for the tail classes only. We'll compute tail logits = input * tail_weight^T
// then log_softmax over that tail's classes, then replace corresponding columns
// in the output tensor.
// This kernel will be called after we have head log_probs and tail logits.
// We'll create a kernel that takes tail logits and output, and replaces columns [start,end)
// with the log_softmax of tail logits.
// ------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void replace_tail_log_probs_kernel(const float* __restrict__ tail_logits,
                                               float* __restrict__ output,
                                               int rows, int tail_size, int start_col) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = tail_logits + row * tail_size;
    float* out_ptr = output + row * (output_cols) + start_col; // output_cols is global n_classes

    // This kernel needs output_cols, but we don't have it as template. We can pass as parameter.
    // We'll compute log_softmax over tail_size and write to out_ptr.
    // Similar to log_softmax but writing to a slice.
    // We'll implement inline.
}

// We'll instead implement a combined kernel for tail replacement that computes
// log_softmax directly and writes into output slice.
// We'll pass output_stride (n_classes) and start_col.
__global__ void tail_log_softmax_write_kernel(const float* __restrict__ logits,
                                              float* __restrict__ output,
                                              int rows, int cols_logits, int start_col,
                                              int output_cols) {
    extern __shared__ float shared[];
    float* sdata = shared;
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = logits + row * cols_logits;
    float* out_ptr = output + row * output_cols + start_col;

    // Find max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < cols_logits; i += blockDim.x) {
        max_val = fmaxf(max_val, row_ptr[i]);
    }
    // Reduction within block
    __shared__ float block_max[256]; // assume blockDim.x <= 256
    block_max[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_max[threadIdx.x] = fmaxf(block_max[threadIdx.x], block_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = block_max[0];
    __syncthreads();

    // Compute sum exp
    float sum_val = 0.0f;
    for (int i = threadIdx.x; i < cols_logits; i += blockDim.x) {
        sum_val += __expf(row_ptr[i] - max_val);
    }
    __shared__ float block_sum[256];
    block_sum[threadIdx.x] = sum_val;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum_val = block_sum[0];
    __syncthreads();

    float log_sum = logf(sum_val);

    // Write log probabilities to the output slice
    for (int i = threadIdx.x; i < cols_logits; i += blockDim.x) {
        out_ptr[i] = (row_ptr[i] - max_val) - log_sum;
    }
}

// ------------------------------------------------------------
// Kernel: assign_tail_indices (optional) - not needed if we compute all samples
// ------------------------------------------------------------

// ------------------------------------------------------------
// Launcher for adaptive_log_softmax_log_prob
// This function orchestrates the entire computation.
// Inputs (all device pointers):
//   input: [batch_size, in_features] float32
//   head_weight: [head_size, in_features] float32
//   head_bias: [head_size] float32
//   tail0_weight0: [tail0_hidden, in_features]
//   tail0_weight1: [tail0_size, tail0_hidden]
//   tail1_weight0: [tail1_hidden, in_features]
//   tail1_weight1: [tail1_size, tail1_hidden]
//   output: allocated by binding, [batch_size, n_classes]
//   cutoffs: int array on device? Actually cutoffs are known at runtime.
//   Since we don't have cutoffs as parameter (contract says cutoffs is int but actually
//   model_new.py passes self.adaptive_log_softmax.cutoffs which is a list).
//   The binding will need to handle this. We'll accept cutoff0 and cutoff1 as ints.
//   Also n_classes, head_size, tail0_size, tail0_hidden, tail1_size, tail1_hidden,
//   and batch_size, in_features.
// ------------------------------------------------------------
extern "C" void adaptive_log_softmax_log_prob_launcher(
    const float* input,
    const float* head_weight,
    const float* head_bias,
    const float* tail0_weight0,
    const float* tail0_weight1,
    const float* tail1_weight0,
    const float* tail1_weight1,
    float* output,
    int batch_size,
    int in_features,
    int head_size,
    int tail0_hidden,
    int tail0_size,
    int tail1_hidden,
    int tail1_size,
    int n_classes,
    int cutoff0,
    int cutoff1,
    cudaStream_t stream) {

    // Temporary buffers for head logits and tail logits
    // We'll allocate device memory using cudaMallocAsync or regular cudaMalloc.
    float* head_logits;
    float* tail0_logits;
    float* tail1_logits;
    cudaMalloc(&head_logits, batch_size * head_size * sizeof(float));
    cudaMalloc(&tail0_logits, batch_size * tail0_size * sizeof(float));
    cudaMalloc(&tail1_logits, batch_size * tail1_size * sizeof(float));

    // 1. Compute head logits = input * head_weight^T + head_bias
    gemm_nt(input, head_weight, head_logits, batch_size, head_size, in_features, stream);
    // Add bias: we can add via a simple kernel or integrate into log_softmax kernel.
    // We'll create a kernel to add bias to head_logits.
    // Alternatively, we can modify log_softmax to take bias? Better to keep separate.
    // We'll launch add_bias kernel.
    dim3 block_add(256);
    dim3 grid_add((head_size + 255) / 256, batch_size);
    add_bias_kernel<<<grid_add, block_add, 0, stream>>>(head_logits, head_bias, batch_size, head_size);

    // 2. Compute head log_probs
    {
        int block_size = 256;
        int shared_mem = block_size * sizeof(float) * 2; // for max and sum reductions
        log_softmax_kernel<256><<<batch_size, block_size, shared_mem, stream>>>(head_logits, output, batch_size, head_size);
    }

    // 3. Determine which samples go to which tail based on argmax of head_logits.
    // We need to find argmax for each row. We can compute in a kernel and store tail assignment
    // in an int array, or we can just compute all tail logits and then use a mask.
    // Since tail computations are independent, we can compute tail0 and tail1 logits for all rows
    // and then replace the relevant columns for rows that belong to that tail.
    // To decide rows, we need argmax of head_logits. We'll compute argmax in a kernel and store
    // in a device array of int per sample.
    int* argmax_head;
    cudaMalloc(&argmax_head, batch_size * sizeof(int));
    compute_argmax_kernel<<<(batch_size + 255)/256, 256, 0, stream>>>(head_logits, argmax_head, batch_size, head_size);

    // 4. Compute tail0 logits: input * tail0_weight0^T -> [batch_size, tail0_hidden]
    float* tail0_hidden_act;
    cudaMalloc(&tail0_hidden_act, batch_size * tail0_hidden * sizeof(float));
    gemm_nt(input, tail0_weight0, tail0_hidden_act, batch_size, tail0_hidden, in_features, stream);
    // Apply ReLU? In AdaptiveLogSoftmaxWithLoss, tail is Sequential with Linear and ReLU?
    // In PyTorch implementation, tail modules are Sequential(Linear, ReLU, Linear).
    // Check model.py: self.adaptive_log_softmax.tail[0] = Sequential(
    //   Linear(in_features, in_features // 4, bias=False),
    //   Linear(in_features // 4, cutoffs[0] - cutoffs[1], bias=False)
    // ) but in this case tail[0] is Sequential(Linear, ReLU, Linear)?
    // Looking at model.py: nn.AdaptiveLogSoftmaxWithLoss creates tail modules with relu between linears.
    // Actually, default AdaptiveLogSoftmaxWithLoss tail modules are Sequential(
    //   Linear(in_features, in_features // 4, bias=False),
    //   ReLU(),
    //   Linear(in_features // 4, sz, bias=False)
    // ). Yes, there is ReLU between two linears.
    // So we need to apply ReLU after first linear.
    // We'll create a ReLU kernel.
    apply_relu_kernel<<<(batch_size * tail0_hidden + 255)/256, 256, 0, stream>>>(tail0_hidden_act, batch_size * tail0_hidden);

    // Compute tail0 logits: tail0_hidden_act * tail0_weight1^T -> [batch_size, tail0_size]
    gemm_nt(tail0_hidden_act, tail0_weight1, tail0_logits, batch_size, tail0_size, tail0_hidden, stream);

    // 5. Compute tail1 similarly.
    float* tail1_hidden_act;
    cudaMalloc(&tail1_hidden_act, batch_size * tail1_hidden * sizeof(float));
    gemm_nt(input, tail1_weight0, tail1_hidden_act, batch_size, tail1_hidden, in_features, stream);
    apply_relu_kernel<<<(batch_size * tail1_hidden + 255)/256, 256, 0, stream>>>(tail1_hidden_act, batch_size * tail1_hidden);
    gemm_nt(tail1_hidden_act, tail1_weight1, tail1_logits, batch_size, tail1_size, tail1_hidden, stream);

    // 6. For each row, depending on argmax:
    //    if argmax < cutoff0: keep head log_probs as is (already computed).
    //    else if argmax < cutoff1: replace columns [cutoff0, cutoff1) with tail0 log_probs.
    //    else: replace columns [cutoff1, n_classes) with tail1 log_probs.
    // We need to write the tail log_probs into output, but only for rows assigned to that tail.
    // We'll create a kernel that takes argmax_head, tail_logits, output, and writes conditionally.
    // For tail0: start_col = cutoff0, size = tail0_size (which is cutoff1 - cutoff0)
    // For tail1: start_col = cutoff1, size = tail1_size (n_classes - cutoff1)

    // We'll launch two kernels: one for tail0, one for tail1, with a condition on argmax.
    int block_size_tail = 256;
    int shared_mem_tail = block_size_tail * sizeof(float) * 2;
    tail_replace_kernel<<<batch_size, block_size_tail, shared_mem_tail, stream>>>(
        tail0_logits, output, argmax_head, batch_size, tail0_size, cutoff0, n_classes, cutoff0, cutoff1);
    tail_replace_kernel<<<batch_size, block_size_tail, shared_mem_tail, stream>>>(
        tail1_logits, output, argmax_head, batch_size, tail1_size, cutoff1, n_classes, cutoff1, n_classes);

    // Free temporary buffers
    cudaFree(head_logits);
    cudaFree(tail0_logits);
    cudaFree(tail1_logits);
    cudaFree(tail0_hidden_act);
    cudaFree(tail1_hidden_act);
    cudaFree(argmax_head);
}

// ------------------------------------------------------------
// Kernel definitions used above that are not yet defined
// ------------------------------------------------------------

// add_bias_kernel
__global__ void add_bias_kernel(float* __restrict__ data, const float* __restrict__ bias,
                                int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (row < rows && col < cols) {
        data[row * cols + col] += bias[col];
    }
}

// compute_argmax_kernel: each block handles one row, using reduction
__global__ void compute_argmax_kernel(const float* __restrict__ logits, int* __restrict__ argmax,
                                      int rows, int cols) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* row_ptr = logits + row * cols;
    // Each thread loads elements strided by blockDim.x
    float max_val = -INFINITY;
    int max_idx = -1;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = row_ptr[i];
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
    }
    // Store max_val and max_idx to shared
    __shared__ float block_max_val[256];
    __shared__ int block_max_idx[256];
    block_max_val[threadIdx.x] = max_val;
    block_max_idx[threadIdx.x] = max_idx;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (block_max_val[threadIdx.x + s] > block_max_val[threadIdx.x]) {
                block_max_val[threadIdx.x] = block_max_val[threadIdx.x + s];
                block_max_idx[threadIdx.x] = block_max_idx[threadIdx.x + s];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        argmax[row] = block_max_idx[0];
    }
}

// apply_relu_kernel
__global__ void apply_relu_kernel(float* __restrict__ data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = fmaxf(data[idx], 0.0f);
    }
}

// tail_replace_kernel: computes log_softmax over tail logits and writes to output
// if argmax condition is satisfied.
__global__ void tail_replace_kernel(const float* __restrict__ tail_logits,
                                    float* __restrict__ output,
                                    const int* __restrict__ argmax_head,
                                    int rows, int tail_size, int start_col, int output_cols,
                                    int lower_bound, int upper_bound) {
    extern __shared__ float shared[];
    int row = blockIdx.x;
    if (row >= rows) return;

    int head_argmax = argmax_head[row];
    // Check if this row belongs to this tail cluster
    if (head_argmax < lower_bound || head_argmax >= upper_bound) {
        return; // do nothing
    }

    const float* row_ptr = tail_logits + row * tail_size;
    float* out_ptr = output + row * output_cols + start_col;

    // Find max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < tail_size; i += blockDim.x) {
        max_val = fmaxf(max_val, row_ptr[i]);
    }
    __shared__ float block_max[256];
    block_max[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_max[threadIdx.x] = fmaxf(block_max[threadIdx.x], block_max[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = block_max[0];
    __syncthreads();

    // Compute sum exp
    float sum_val = 0.0f;
    for (int i = threadIdx.x; i < tail_size; i += blockDim.x) {
        sum_val += __expf(row_ptr[i] - max_val);
    }
    __shared__ float block_sum[256];
    block_sum[threadIdx.x] = sum_val;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            block_sum[threadIdx.x] += block_sum[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum_val = block_sum[0];
    __syncthreads();

    float log_sum = logf(sum_val);

    // Write log probabilities
    for (int i = threadIdx.x; i < tail_size; i += blockDim.x) {
        out_ptr[i] = (row_ptr[i] - max_val) - log_sum;
    }
}

// ------------------------------------------------------------
// End of kernels
// ------------------------------------------------------------
