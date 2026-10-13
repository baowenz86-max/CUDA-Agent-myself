#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>

// ------------------------- Helper: error checking -------------------------
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                          \
      abort();                                                                   \
    }                                                                           \
  } while (0)

// ------------------------- Linear (cuBLAS) -------------------------
// input: [batch_size, in_features], weight: [out_features, in_features]
// Computes output = input @ weight^T + bias.
// Uses cuBLAS column-major: A = weight (out x in), B = input (batch x in) as column-major
//   -> A^T * B gives (in x out) * (batch x in) but not correct.
// Alternative: use cublasSgemm with transb=CUBLAS_OP_T on input:
//   A (out x in) * B^T (in x batch) => out x batch, then transpose in kernel? not efficient.
// Better: treat weight as column-major already: weight stored row-major [out, in] means column-major [in, out].
//   So A = weight (column-major) is (in x out). Need output (out x batch) column-major.
//   C = A * B where B = input (column-major) is (in x batch).
//   So cublasSgemm with opA=CUBLAS_OP_N, opB=CUBLAS_OP_N, A: in x out, B: in x batch -> C: in x out? not good.
// Actually, standard cuBLAS is column-major, data pointer points to first element, dimensions are m,n,k.
// If we want C[out,batch] in column-major = row-major output[batch,out] with row stride out.
// Let's use cublasSgemm with:
//   A = weight, m = out_features, n = batch_size, k = in_features, leading dimension lda = in_features (row-major)
//   B = input, m = in_features, n = batch_size, k = in_features (but B is row-major input), we want B^T.
// Better approach: Use cublasSgemm with CUBLAS_OP_T for B, transpose input as needed.
// Standard: output = input @ weight.T => in row-major: C[batch,out] = sum_k input[batch,k] * weight[out,k]
// In column-major, that is A = weight as column-major (in,out), B = input as column-major (batch,in) but we want B^T.
// So use cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, batch_size, in_features, &alpha, weight, in_features, input, in_features, &beta, output, out_features)
// This yields C^T (out x batch) = B^T (in x batch) * A (out x in) => wait: cublasSgemm computes C = op(A) * op(B) + beta*C with dimensions m,n,k.
// Set op(A) = A = weight (T? Actually we want C = weight * input^T? Let's test with m=out, n=batch, k=in: A is m x k = out x in, B is k x n = in x batch.
// We want A = weight (row-major out x in) and B = input^T (in x batch). Since input is row-major batch x in, input^T is column-major in x batch.
// So set op(B) = CUBLAS_OP_T? No, B is input (batch x in). We want B^T (in x batch). In column-major storage, input is row-major batch x in, which is the transpose of column-major in x batch.
// So if we pass input pointer with leading dimension in_features, cuBLAS interprets it as column-major with dimensions (in_features, batch_size) if we set m=in, n=batch? That is B is in x batch column-major, which is exactly input^T. But we need B^T? Let's step back.
// We want C = input @ weight.T (batch x in) * (in x out) = batch x out.
// cublasSgemm computes C = op(A) * op(B) where A is m x k, B is k x n, C is m x n.
// We want m = batch, n = out, k = in.
// Then A should be input (batch x in), B should be weight.T (in x out).
// A = input row-major is column-major with m=batch, k=in? Actually if we treat A as column-major, the storage is column-major, but A is batch x in with leading dimension batch? Row-major means contiguous rows, so A_{i,j} = input[i*in + j]. Column-major would be A_{i,j} = input[j*batch + i], which is not stored. So we need to use op(A) = CUBLAS_OP_T to transpose the input as it is stored (row-major -> column-major).
// For B we want weight.T (in x out). Weight is row-major out x in, stored as weight_{i,j} = weight[i*in + j]. Column-major of weight.T would be B_{i,j} = weight.T[j*out + i]?  Actually if we want B = weight.T in row-major (in x out), its column-major storage is the transpose, which equals weight row-major (out x in). So we can pass weight with op(B) = CUBLAS_OP_N and it will be interpreted as column-major B (in x out) with leading dimension out? Hmm.
// This is known trick: to compute C = A * B^T with A row-major, use cublasSgemm with transa=CUBLAS_OP_N, transb=CUBLAS_OP_T, m=n_rows_A, n=n_rows_B, k=n_cols_A.
// So set m = batch_size, n = out_features, k = in_features, A = input, lda = in_features, transa=CUBLAS_OP_N, B = weight, ldb = in_features (since weight row-major out x in, but B is interpreted as k x n = in x out, so leading dimension = in_features? Actually B is k x n = in x out, in column-major storage, the leading dimension of B is ld = in_features? No, column-major means column length is k=in, so ld must be at least in_features. The data is weight row-major with row stride in_features, so B[i,j] = weight[j*in_features + i]? That is correct: for column-major B of dimensions (in x out), B[i,j] stored at j*ld + i, with ld = in_features. That matches weight row-major. So yes, transb=CUBLAS_OP_N.
// But we want B = weight^T? Wait, we want C = input * weight^T, B should be weight^T (in x out). But if we pass weight as B with transb=N, B is in x out column-major, which is exactly weight^T stored in column-major. So correct.
// So cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, batch_size, out_features, in_features, &alpha, input, in_features, weight, in_features, &beta, output, batch_size)
// And output must be column-major with m=batch, n=out, ld=out? Wait output C is m x n = batch x out. In column-major, ld must be at least m = batch. But output row-major would have ld = out. To avoid transpose, we can instead output to a temporary column-major buffer and then transpose? That wastes time.
// Better: use cublasSgemm with transb=CUBLAS_OP_T and A=weight as column-major, B=input as column-major? Let's reconsider.
// We want C = weight (out x in) * input^T (in x batch) => out x batch. Then output can be transposed to get batch x out.
// But maybe we can directly use cuBLAS with proper leading dimensions.
// Many implementations use cublasSgemm with transa=CUBLAS_OP_T, transb=CUBLAS_OP_N, m=out, n=batch, k=in, A=weight (out x in) with ld=in (row-major), B=input (batch x in) with ld=in (row-major), C=output (out x batch) with ld=out? That gives C = weight^T * input? Hmm.
// Let's just implement a custom linear kernel that is simple and efficient enough for this size (128x768 @ 768x200). Grid with each thread computing one output element, using shared memory tiling to reuse input and weight. But we can also use cuBLAS with a temporary transpose or use cublasSgemm with reinterpretation.
// Given performance requirement, cuBLAS is likely faster. We'll use cublasSgemm with op(A)=CUBLAS_OP_N, op(B)=CUBLAS_OP_N, but we need to pass weight as column-major of dimensions (in x out). Then output is batch x out column-major, which is actually row-major output with stride out? Let's verify.
// If we set m=batch, n=out, k=in, A=input (batch x in) row-major but treated as column-major with ld=batch? We can't because row stride is in. So use transA=CUBLAS_OP_T: A = input^T (in x batch) column-major, stored as input row-major? If we pass input pointer and set transA=CUBLAS_OP_T, cuBLAS will interpret A as column-major in x batch with ldA = in_features? Actually for op(A)=T, A is originally k x m in column-major, so we need A stored as column-major k x m. input row-major is batch x in, which is column-major in x batch with ld = batch (if we set k=in, m=batch). So if we set transA=CUBLAS_OP_T, m_output = batch, k=in, A input pointer with lda=batch? But input row-major has row stride in, not batch. This is getting messy.
// Simpler: write a custom kernel for linear for float32 that is reasonably optimized. Given batch=128, in=768, out=200, each output element requires 768 multiplications. Total 128*200*768 ~ 19.6M multiplications. A simple kernel with one thread per output element may be okay but not optimal. However, for this task, correctness and passing verification is crucial, performance optimization can come later. But the task says to implement complete, compilable non-conv kernels. We can use cuBLAS for GEMM; it's allowed and recommended for performance. We need to get the parameters right.
// Let's derive the correct call: We want C = A @ B where A is batch x in (row-major), B is in x out (row-major) and C is batch x out (row-major).
// cublasSgemm treats all matrices as column-major. If we want row-major C, we can compute C^T = B^T @ A^T, which is out x batch. Then we can store C^T column-major, which is C row-major.
// So set m = out_features, n = batch_size, k = in_features.
// op(A) should be B^T = weight^T? Wait B^T is (out x in). Let's denote:
//   We want C_row = A_row (batch x in) * B_row (in x out).
//   Transpose: C_row = (B_row^T * A_row^T)^T.
//   So in column-major, we compute C_col = B_row^T * A_row^T where C_col is C_row stored column-major (which is the same memory as row-major C).
//   In cuBLAS: m = out, n = batch, k = in.
//   A_matrix (m x k) = B_row^T (out x in). But B_row is weight (in x out) row-major. Its transpose B_row^T is out x in. In column-major storage, B_row^T stored as column-major with ld = in_features (since B_row row-major has row stride in_features). So we can pass weight pointer to A with transA=CUBLAS_OP_N, lda = in_features. Wait A is m x k = out x in, column-major means ld = m? Column-major ld >= m. We have weight stored row-major out x in with row stride in_features. If we interpret weight as column-major out x in, the element (i,j) should be at j*ld + i, with ld at least out. But we have weight[i*in + j], so if we set ld = in, we get column-major matrix of dimensions in x out? Let's see: if we set ld = in_features, and we access A[i,j] = weight[j*ld + i] = weight[j*in + i]. This corresponds to matrix with dimensions (in x out) because i ranges 0..in-1, j ranges 0..out-1. So it's a (in x out) column-major matrix. That matches B_row^T? B_row is in x out, so B_row^T is out x in. Our matrix is in x out, not out x in. So it's B_row^T? No, B_row^T is out x in. Our matrix is in x out. So it's not B_row^T. Actually, B_row (in x out) in column-major would be out x in? Hmm.
// Let's not overcomplicate. Use a custom kernel with shared memory tiling: each block computes a tile of output, using shared memory for input and weight tiles. This avoids cuBLAS complexity and is acceptable for this moderate size. We'll implement robust kernel.

// ------------------------- Linear custom kernel -------------------------
// One block computes a TILE_M x TILE_N tile of output.
// TILE_M = 32 (batch dimension), TILE_N = 32 (out dimension), TILE_K = 32.
// Each thread computes multiple output elements.
__global__ void linear_kernel(const float* __restrict__ input,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ output,
                              int batch_size, int in_features, int out_features,
                              bool has_bias) {
    __shared__ float s_input[32][32];
    __shared__ float s_weight[32][32];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.x * 32 + ty;
    int col = blockIdx.y * 32 + tx;
    float acc = 0.0f;
    for (int k = 0; k < in_features; k += 32) {
        // load input tile: [row, k + tx]
        int kk = k + tx;
        if (row < batch_size && kk < in_features) {
            s_input[ty][tx] = input[row * in_features + kk];
        } else {
            s_input[ty][tx] = 0.0f;
        }
        // load weight tile: [col, k + ty]
        int weight_row = col;
        int weight_col = k + ty;
        if (weight_row < out_features && weight_col < in_features) {
            s_weight[ty][tx] = weight[weight_row * in_features + weight_col];
        } else {
            s_weight[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int kk2 = 0; kk2 < 32; ++kk2) {
            acc += s_input[ty][kk2] * s_weight[tx][kk2];
        }
        __syncthreads();
    }
    if (row < batch_size && col < out_features) {
        float val = acc + (has_bias ? bias[col] : 0.0f);
        output[row * out_features + col] = val;
    }
}

// Launcher for linear
void linear_forward_launcher(const float* input, const float* weight, const float* bias,
                             float* output, int batch_size, int in_features, int out_features,
                             bool has_bias, cudaStream_t stream) {
    dim3 block(32, 32);
    dim3 grid((out_features + 31) / 32, (batch_size + 31) / 32);
    linear_kernel<<<grid, block, 0, stream>>>(input, weight, bias, output,
                                              batch_size, in_features, out_features, has_bias);
    CUDA_CHECK(cudaGetLastError());
}

// ------------------------- LogSoftmax kernel -------------------------
__global__ void log_softmax_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int batch_size, int num_classes) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    // shared max and sum
    __shared__ float s_max;
    __shared__ float s_sum;
    if (tid == 0) {
        s_max = -FLT_MAX;
        s_sum = 0.0f;
    }
    __syncthreads();
    // find max in row
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float v = input[row * num_classes + i];
        if (v > s_max) {
            // atomic max via compare? Use atomicMax for float as int? simpler: loop over few threads.
            // To avoid race, use reduction. Since num_classes=200, small, we can use one thread to compute max.
            // For simplicity and correctness, we can use a single thread per row to do all work.
        }
    }
    // Since num_classes=200, one thread per row is fine. We'll use one block per row with 128 threads and reduction.
    // but for simplicity, use one thread per row? That may be slow but correct. We'll use 256 threads per row, each handles a few elements.
    // Actually, implement robust reduction with shared memory.
}

// Given small dims (200), we can use one thread per row to compute log_softmax, but that underutilizes GPU.
// For better performance, use one block per row and 256 threads, each handles a subset, then reduce in shared memory.
// We'll implement a generic softmax/log_softmax kernel with reduction.

// ------------------------- Softmax kernel -------------------------
__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int batch_size, int num_classes) {
    int row = blockIdx.x;
    float* sdata = SharedMemory<float>(); // not available, we'll use extern __shared__
}

// Use simpler approach: one thread per row using a loop. Since num_classes=200 and batch=128, total 128 threads, which is small but acceptable.
// But that may be slower than baseline. For correctness first, we can use one block per row with 256 threads and reduction, but it's more code.
// Let's write a generic softmax using the softmax decomposition with online softmax to avoid multiple passes, but we need two passes for log_softmax and softmax.
// For log_softmax, we need max and sum. For softmax, we need max and sum.
// We can combine reduction with shared memory.

// We'll implement a simple vectorized approach using a single block per row.
const int ROW_THREADS = 256;

__global__ void log_softmax_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int batch_size, int num_classes) {
    int row = blockIdx.x;
    __shared__ float s_max;
    __shared__ float s_sum;
    // thread id within block
    int tid = threadIdx.x;
    // initialize shared
    if (tid == 0) {
        s_max = -FLT_MAX;
        s_sum = 0.0f;
    }
    __syncthreads();
    // find max
    float local_max = -FLT_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float v = input[row * num_classes + i];
        if (v > local_max) local_max = v;
    }
    // reduce max across threads (block reduction)
    __shared__ float s_max_reduce[ROW_THREADS];
    s_max_reduce[tid] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other = s_max_reduce[tid + offset];
            if (other > s_max_reduce[tid]) s_max_reduce[tid] = other;
        }
        __syncthreads();
    }
    if (tid == 0) s_max = s_max_reduce[0];
    __syncthreads();
    float max_val = s_max;
    // compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        local_sum += expf(input[row * num_classes + i] - max_val);
    }
    __shared__ float s_sum_reduce[ROW_THREADS];
    s_sum_reduce[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum_reduce[tid] += s_sum_reduce[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) s_sum = s_sum_reduce[0];
    __syncthreads();
    float sum_val = s_sum;
    float log_sum = logf(sum_val);
    // write output: x - max - log(sum)
    for (int i = tid; i < num_classes; i += blockDim.x) {
        output[row * num_classes + i] = input[row * num_classes + i] - max_val - log_sum;
    }
}

__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int batch_size, int num_classes) {
    int row = blockIdx.x;
    __shared__ float s_max;
    __shared__ float s_sum;
    int tid = threadIdx.x;
    if (tid == 0) {
        s_max = -FLT_MAX;
        s_sum = 0.0f;
    }
    __syncthreads();
    float local_max = -FLT_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float v = input[row * num_classes + i];
        if (v > local_max) local_max = v;
    }
    __shared__ float s_max_reduce[ROW_THREADS];
    s_max_reduce[tid] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other = s_max_reduce[tid + offset];
            if (other > s_max_reduce[tid]) s_max_reduce[tid] = other;
        }
        __syncthreads();
    }
    if (tid == 0) s_max = s_max_reduce[0];
    __syncthreads();
    float max_val = s_max;
    float local_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        local_sum += expf(input[row * num_classes + i] - max_val);
    }
    __shared__ float s_sum_reduce[ROW_THREADS];
    s_sum_reduce[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum_reduce[tid] += s_sum_reduce[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) s_sum = s_sum_reduce[0];
    __syncthreads();
    float sum_val = s_sum;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        output[row * num_classes + i] = expf(input[row * num_classes + i] - max_val) / sum_val;
    }
}

// Launchers for softmax/log_softmax
void log_softmax_launcher(const float* input, float* output, int batch_size, int num_classes, cudaStream_t stream) {
    int threads = 256;
    int blocks = batch_size;
    log_softmax_kernel<<<blocks, threads, 0, stream>>>(input, output, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
}

void softmax_launcher(const float* input, float* output, int batch_size, int num_classes, cudaStream_t stream) {
    int threads = 256;
    int blocks = batch_size;
    softmax_kernel<<<blocks, threads, 0, stream>>>(input, output, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
}

// ------------------------- KL Div batchmean -------------------------
// Compute sum over all elements of target_probs * (log(target_probs) - log_probs), then divide by batch_size.
// We'll use one block per batch row, accumulate row sum using reduction, then atomicAdd to a temporary global sum.
// To avoid multiple kernels, we can use a two-pass approach: first kernel computes per-row sums, second kernel sums rows and divides.
// For simplicity, use one kernel with atomicAdd to a global float (initialized to 0) and then divide by batch_size in a second tiny kernel or using cudaMemcpy? But output is a scalar tensor, we can write directly using a single block after atomicAdd? The atomicAdd introduces non-determinism but for sum, order doesn't affect precision significantly (maybe slight). We can use a separate kernel to compute final scalar from a global sum.
// We'll allocate a temporary device memory for sum, launch kernel to compute per-row partial sums with atomicAdd, then launch a single-thread kernel to read sum and divide by batch_size, writing to output.
// Or we can have each block compute row sum and then use atomicAdd to a scalar output directly if we divide by batch_size in the same kernel? But each block would divide its own sum and atomicAdd, causing each block to add (row_sum/batch_size), which is not the same as sum/batch_size due to floating point rounding. We want exact sum/batch_size. So better to accumulate sum first, then divide.
// We'll use a global float* partial_sums array with one entry per row, but we still need to sum across rows. Use second kernel.
// Since batch_size=128, can use a single block to sum all rows after computing per-row sums.
// Approach: Kernel 1: each block computes row sum and stores in partial_sums[row]. Kernel 2: single block reduces partial_sums to output[0] = sum / batch_size.
__global__ void kl_div_rowsum_kernel(const float* __restrict__ log_probs,
                                     const float* __restrict__ target_probs,
                                     float* __restrict__ partial_sums,
                                     int batch_size, int num_classes) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float s_sum[256];
    float local_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        float lp = log_probs[row * num_classes + i];
        float tp = target_probs[row * num_classes + i];
        // KLDivLoss expectation target * (log(target) - input), but input is log_probs already.
        // F.kl_div expects input as log-probabilities, target as probabilities.
        // So elementwise: target * (log(target) - log_probs)
        // Since target is softmax output (>=0, sum to 1), log(target) is valid as target>0 (softmax never exactly 0 for finite inputs, but could be 0 if underflow? For float, possible but not with typical random, but we can guard: if tp > 0, add tp * (logf(tp) - lp); else add 0. PyTorch likely treats 0*log(0)=0.
        if (tp > 0.0f) {
            local_sum += tp * (logf(tp) - lp);
        }
    }
    s_sum[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_sums[row] = s_sum[0];
    }
}

__global__ void kl_div_final_kernel(const float* __restrict__ partial_sums,
                                    float* __restrict__ output,
                                    int batch_size) {
    // single thread
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            total += partial_sums[i];
        }
        output[0] = total / batch_size;
    }
}

void kl_div_batchmean_launcher(const float* log_probs, const float* target_probs,
                               float* output, int batch_size, int num_classes,
                               float* partial_sums, cudaStream_t stream) {
    int threads = 256;
    kl_div_rowsum_kernel<<<batch_size, threads, 0, stream>>>(log_probs, target_probs, partial_sums, batch_size, num_classes);
    CUDA_CHECK(cudaGetLastError());
    kl_div_final_kernel<<<1, 1, 0, stream>>>(partial_sums, output, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

// ------------------------- L1 loss sum params -------------------------
// For each i in [0, num_params), compute mean over batch of |x[b,i] - params[i]|, then sum over i.
// We can use one block per column, reduce over batch, then add to total via atomicAdd or second kernel.
// batch_size=128, num_params=200, so we can have each block compute column sum and atomicAdd to a global sum, then divide by batch_size per column? Actually mean per column = sum / batch_size, then total sum of means = (sum over i of column_sum_i) / batch_size. So we can compute total sum of column sums then divide by batch_size once.
// So compute overall sum of |x[b,i] - params[i]| over all b and i, then divide by batch_size.
// This matches sum of means = (1/batch_size) * sum_i sum_b |...|.
// We can use a single kernel with atomicAdd to a global float, but again precision. We can use two-stage reduction: each block computes partial sum for its column and atomicAdd to a global sum (float), then a final kernel divides by batch_size.
// Or use a grid-stride loop over all elements with atomicAdd. But atomicAdd on float is fine and deterministic enough for verification tolerance.
// We'll use a kernel with grid as number of columns, each block reduces over batch, then atomicAdd to a global float (initialized to 0), then final kernel divides.
__global__ void l1_loss_column_sum_kernel(const float* __restrict__ x,
                                          const float* __restrict__ params,
                                          float* __restrict__ global_sum,
                                          int batch_size, int num_params) {
    int col = blockIdx.x;
    int tid = threadIdx.x;
    __shared__ float s_sum[256];
    float local_sum = 0.0f;
    float p = params[col];
    for (int b = tid; b < batch_size; b += blockDim.x) {
        local_sum += fabsf(x[b * num_params + col] - p);
    }
    s_sum[tid] = local_sum;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(global_sum, s_sum[0]);
    }
}

__global__ void l1_loss_final_kernel(const float* __restrict__ global_sum,
                                     float* __restrict__ output,
                                     int batch_size) {
    if (threadIdx.x == 0) {
        output[0] = *global_sum / batch_size;
    }
}

void l1_loss_sum_params_launcher(const float* x, const float* params,
                                 float* output, int batch_size, int num_params,
                                 float* global_sum, cudaStream_t stream) {
    int threads = 256;
    // Initialize global_sum to 0 before kernel
    cudaMemsetAsync(global_sum, 0, sizeof(float), stream);
    l1_loss_column_sum_kernel<<<num_params, threads, 0, stream>>>(x, params, global_sum, batch_size, num_params);
    CUDA_CHECK(cudaGetLastError());
    l1_loss_final_kernel<<<1, 1, 0, stream>>>(global_sum, output, batch_size);
    CUDA_CHECK(cudaGetLastError());
}

// ------------------------- Extern C launcher declarations -------------------------
// These will be called from binding.cpp.
extern "C" {

void linear_forward_launcher_cuda(const float* input, const float* weight, const float* bias,
                                  float* output, int batch_size, int in_features, int out_features,
                                  bool has_bias, cudaStream_t stream) {
    linear_forward_launcher(input, weight, bias, output, batch_size, in_features, out_features, has_bias, stream);
}

void log_softmax_launcher_cuda(const float* input, float* output, int batch_size, int num_classes, cudaStream_t stream) {
    log_softmax_launcher(input, output, batch_size, num_classes, stream);
}

void softmax_launcher_cuda(const float* input, float* output, int batch_size, int num_classes, cudaStream_t stream) {
    softmax_launcher(input, output, batch_size, num_classes, stream);
}

void kl_div_batchmean_launcher_cuda(const float* log_probs, const float* target_probs,
                                    float* output, int batch_size, int num_classes,
                                    float* partial_sums, cudaStream_t stream) {
    kl_div_batchmean_launcher(log_probs, target_probs, output, batch_size, num_classes, partial_sums, stream);
}

void l1_loss_sum_params_launcher_cuda(const float* x, const float* params,
                                      float* output, int batch_size, int num_params,
                                      float* global_sum, cudaStream_t stream) {
    l1_loss_sum_params_launcher(x, params, output, batch_size, num_params, global_sum, stream);
}

} // extern "C"