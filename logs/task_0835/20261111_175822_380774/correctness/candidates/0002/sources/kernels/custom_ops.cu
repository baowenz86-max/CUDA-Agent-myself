// custom_ops.cu
// CUDA kernels and launchers for multihead attention and log10_abs_eps
// No torch/ATen dependencies; only CUDA runtime and cuBLAS required.

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// Helper macro for CUDA error checking in launchers
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            // In launcher we cannot throw, but we can print and return? \
            // For production we might want to propagate errors, but here we ignore to keep signature simple. \
            // The binding layer should check for errors after launcher returns. \
            // We'll still print to stderr. \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while (0)

// ========================== Element-wise ops ==========================

__global__ void log10_abs_eps_kernel(const float* __restrict__ x, float* __restrict__ out, long long numel, float eps) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        out[i] = log10f(fabsf(x[i]) + eps);
    }
}

// ========================== GEMM helpers using cuBLAS ==========================

// Utility to perform C = alpha * A * B + beta * C, where A is row-major but cuBLAS expects column-major.
// We'll use cublasSgemm with transposes appropriately.
static void sgemm_rowmajor(cublasHandle_t handle, bool transA, bool transB,
                           int m, int n, int k,
                           float alpha, const float* A, int lda,
                           const float* B, int ldb,
                           float beta, float* C, int ldc,
                           cudaStream_t stream) {
    cublasSetStream(handle, stream);
    // cuBLAS uses column-major, but our matrices are row-major.
    // We can treat C = A * B as column-major by using transpositions and swapping m,n appropriately.
    // Better: use cublasSgemm with CUBLAS_OP_T on A and B to compute C^T = B^T * A^T.
    // But we need C (row-major) = A (row-major) * B (row-major).
    // Equivalent to C^T = B^T * A^T in column-major. So we set:
    // cublasSgemm(handle, CUBLAS_OP_T (on B?), ...). Let's derive:
    // We want C = A * B, all row-major, dimensions: A [m x k], B [k x n], C [m x n].
    // In column-major representation, these are A_col [k x m], B_col [n x k], C_col [n x m].
    // C_col = B_col * A_col (column-major multiplication) -> C_col [n x m].
    // So we need to compute C_col = B_col * A_col. B_col has dimensions n x k, A_col k x m.
    // Therefore, call cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
    //                           n, m, k, alpha, B_col, n, A_col, k, beta, C_col, n).
    // Wait: In column-major, B_col is stored as B in column-major order, but B is actually row-major with dimensions k x n.
    // The column-major view of B has dimensions n x k and pointer B. Similarly A_col has dimensions k x m, pointer A.
    // C_col has dimensions n x m, pointer C.
    // So call: cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, B, n, A, k, &beta, C, n).
    // Let's test with simple 2x2.
    // Actually this works if we treat row-major as column-major of transposed. So use OP_N.
    // But if we want to apply transA/transB, we need to adjust. For simplicity, we'll only support non-transposed cases for projection and attention.
    // For in_proj: X [L*B, E] * W^T [E, 3E] -> QKV [L*B, 3E]. We can use row-major: compute X * W^T directly by setting transB = CUBLAS_OP_T? Since W is stored [3E, E], we want W^T [E, 3E]. We can use cublasSgemm with transB = CUBLAS_OP_T on W.
    // We'll implement a more general helper using transA/transB as bools.
    // For row-major, if transB is true, we pass B and ldb, use CUBLAS_OP_T, else CUBLAS_OP_N.
    // This becomes tricky. Instead, we can manually set up cuBLAS calls in each launcher as needed.
    // We'll break into smaller functions for each linear operation.
    // For now, we'll implement linear via a custom kernel? But GEMM kernels are complex and slow. Better use cuBLAS.
    // We'll write explicit cublasSgemm calls with appropriate transposes.
    // We'll create a function for each specific multiplication in multihead_attn_forward.
    cublasStatus_t status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasSgemm failed\n");
    }
}

// ========================== Multihead Attention Launcher ==========================

// Launcher signature:
// Inputs: query, key, value pointers (float), in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias
// Dimensions: batch, seq_len, embed_dim, num_heads
// batch_first bool
// Outputs: attn_output pointer, attn_output_weights pointer
// All tensors are assumed to be float32 and on CUDA.
// The launcher is responsible for all computations and writes to output pointers.

extern "C" void multihead_attn_forward(
    const float* query, const float* key, const float* value,
    const float* in_proj_weight, const float* in_proj_bias,
    const float* out_proj_weight, const float* out_proj_bias,
    int64_t batch_size, int64_t seq_len, int64_t embed_dim, int64_t num_heads,
    bool batch_first,
    float* attn_output, float* attn_output_weights,
    cudaStream_t stream)
{
    // Input checks: embed_dim must be divisible by num_heads
    if (embed_dim % num_heads != 0) {
        fprintf(stderr, "embed_dim must be divisible by num_heads\n");
        return;
    }
    int64_t head_dim = embed_dim / num_heads;

    // cuBLAS handle
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasCreate failed\n");
        return;
    }
    cublasSetStream(handle, stream);

    // Determine tensor layouts based on batch_first
    // For batch_first=True: query/key/value shape [batch, seq_len, embed_dim], strides (seq_len*embed_dim, embed_dim, 1)
    // For batch_first=False: shape [seq_len, batch, embed_dim], strides (batch*embed_dim, embed_dim, 1)
    // We'll handle both, but for this task batch_first=True only.
    int64_t batch, seq;
    int64_t query_stride_batch, query_stride_seq, key_stride_batch, key_stride_seq, value_stride_batch, value_stride_seq;
    if (batch_first) {
        batch = batch_size;
        seq = seq_len;
        // Assuming contiguous strides: query stride [seq*embed_dim, embed_dim, 1]
        query_stride_batch = seq * embed_dim;
        query_stride_seq = embed_dim;
        key_stride_batch = seq * embed_dim;
        key_stride_seq = embed_dim;
        value_stride_batch = seq * embed_dim;
        value_stride_seq = embed_dim;
    } else {
        batch = batch_size;
        seq = seq_len;
        // shape [seq, batch, embed_dim]
        query_stride_batch = embed_dim;
        query_stride_seq = batch * embed_dim;
        key_stride_batch = embed_dim;
        key_stride_seq = batch * embed_dim;
        value_stride_batch = embed_dim;
        value_stride_seq = batch * embed_dim;
    }

    // Allocate temporary buffers for Q, K, V, QK^T scores, softmax probabilities, attention output before out_proj
    // Q, K, V: each [batch*seq, embed_dim] (we flatten batch and seq for GEMM)
    // Actually we can compute in_proj for all sequences at once:
    // Input X: [batch*seq, embed_dim] -> in_proj_weight [3*embed_dim, embed_dim]^T -> QKV [batch*seq, 3*embed_dim]
    // But query, key, value are separate tensors, so we need to compute separate projections.
    // For each of q,k,v: X_i [batch*seq, embed_dim] * W_i^T [embed_dim, embed_dim] -> Q_i [batch*seq, embed_dim]
    // We'll do three GEMMs.
    int64_t total_seq = batch * seq;

    // Temporary buffers
    float *d_q, *d_k, *d_v;
    CUDA_CHECK(cudaMalloc(&d_q, total_seq * embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, total_seq * embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, total_seq * embed_dim * sizeof(float)));
    // Buffers for attention weights: shape [batch, num_heads, seq, seq] (we'll compute per head then average)
    // To save memory, we can compute one head at a time and accumulate average weights.
    // For simplicity, allocate full weights: batch * num_heads * seq * seq floats, then average later.
    // But attn_output_weights contract expects shape [batch, seq, seq] (average over heads).
    // We'll compute per-head weights, accumulate sum, then divide by num_heads.
    float *d_weights_sum; // [batch, seq, seq]
    CUDA_CHECK(cudaMalloc(&d_weights_sum, batch * seq * seq * sizeof(float)));
    // Initialize d_weights_sum to zero
    CUDA_CHECK(cudaMemset(d_weights_sum, 0, batch * seq * seq * sizeof(float)));

    // Temporary for QK^T scores: [batch, num_heads, seq, seq] can be large, but for each head we need [batch, seq, seq]
    // We'll allocate a buffer for scores per head (batch*seq*seq), and reuse.
    float *d_scores;
    CUDA_CHECK(cudaMalloc(&d_scores, batch * seq * seq * sizeof(float)));
    // Temporary for softmax probabilities: same size
    float *d_probs;
    CUDA_CHECK(cudaMalloc(&d_probs, batch * seq * seq * sizeof(float)));

    // Temporary for attention output before out_proj: [batch, seq, embed_dim]
    float *d_attn_before_out;
    CUDA_CHECK(cudaMalloc(&d_attn_before_out, batch * seq * embed_dim * sizeof(float)));

    // Step 1: Compute Q = query * W_q^T + b_q, etc.
    // in_proj_weight layout: [3*embed_dim, embed_dim], rows: first embed_dim rows for Q, next for K, next for V.
    // in_proj_bias: length 3*embed_dim.
    const float* W_q = in_proj_weight; // [embed_dim, embed_dim]
    const float* W_k = in_proj_weight + embed_dim * embed_dim;
    const float* W_v = in_proj_weight + 2 * embed_dim * embed_dim;
    const float* b_q = in_proj_bias;
    const float* b_k = in_proj_bias + embed_dim;
    const float* b_v = in_proj_bias + 2 * embed_dim;

    // GEMM for Q: Q = query * W_q^T (W_q is [embed_dim, embed_dim] row-major)
    // query is [batch*seq, embed_dim], row-major. We want C = query * W_q^T [batch*seq, embed_dim].
    // W_q^T is [embed_dim, embed_dim], which can be passed as transB = CUBLAS_OP_T on W_q.
    float alpha = 1.0f, beta = 0.0f;
    // We need to copy non-contiguous query into a contiguous buffer first? We assumed contiguous, but contract says supports strides.
    // Since we are passed raw pointers and strides are not passed, we rely on binding layer providing contiguous data_ptr?
    // Actually launcher receives raw pointers; in binding layer we may need to call .contiguous() if not contiguous.
    // To be safe, we assume inputs are contiguous. The binding layer should ensure that.
    // But contract says "supports strides" meaning input tensors can be non-contiguous, but the launcher only gets raw pointer, so binding layer must pass a contiguous version and then copy output back if needed?
    // Typically, .data_ptr<float>() returns pointer to underlying storage, which may be non-contiguous; but our kernel would access incorrectly.
    // Since contract says supports strides, we should either handle strides in launcher by adding stride parameters, or binding layer makes contiguous copies and then copies back output. However, binding layer is not written yet, and launcher signature is fixed by contract.
    // Looking at custom_ops_contract.json, parameters are tensors with "layout": "supports strides", but launcher only gets pointers. The binding layer can make contiguous copies before calling launcher and then copy back if necessary. That is acceptable.
    // We'll proceed assuming contiguous buffers.
    
    // Use cublasSgemm with transB = CUBLAS_OP_T to compute Q = query * W_q^T.
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                         total_seq, embed_dim, embed_dim,
                         &alpha, query, total_seq, W_q, embed_dim,
                         &beta, d_q, total_seq);
    if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "GEMM Q failed\n"); return; }

    // Add bias to Q
    // bias add kernel
    // We'll implement a simple kernel: q[i*embed_dim + j] += b_q[j]
    {
        dim3 block(256);
        dim3 grid((total_seq * embed_dim + block.x - 1) / block.x);
        // Could improve with grid-stride loop, but simple is fine for now
        add_bias_kernel<<<grid, block, 0, stream>>>(d_q, b_q, total_seq, embed_dim);
    }

    // K, V similarly
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                         total_seq, embed_dim, embed_dim,
                         &alpha, key, total_seq, W_k, embed_dim,
                         &beta, d_k, total_seq);
    if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "GEMM K failed\n"); return; }
    {
        dim3 block(256);
        dim3 grid((total_seq * embed_dim + block.x - 1) / block.x);
        add_bias_kernel<<<grid, block, 0, stream>>>(d_k, b_k, total_seq, embed_dim);
    }

    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                         total_seq, embed_dim, embed_dim,
                         &alpha, value, total_seq, W_v, embed_dim,
                         &beta, d_v, total_seq);
    if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "GEMM V failed\n"); return; }
    {
        dim3 block(256);
        dim3 grid((total_seq * embed_dim + block.x - 1) / block.x);
        add_bias_kernel<<<grid, block, 0, stream>>>(d_v, b_v, total_seq, embed_dim);
    }

    // Step 2: Compute attention for each head and accumulate weights
    // For each head h, compute scores = Q_h * K_h^T / sqrt(head_dim), softmax, then attn_out_h = softmax * V_h
    // Q, K, V are stored interleaved: for each token (batch*seq index), embed_dim = num_heads * head_dim.
    // Token index t = batch * seq + s. For head h, its subspace starts at t*embed_dim + h*head_dim.
    // We'll loop over heads.
    for (int64_t h = 0; h < num_heads; ++h) {
        // Compute scores for all batches: [batch, seq, seq]
        // We can use a custom kernel to compute Q_h * K_h^T for each batch.
        // Alternatively, use GEMM with strided batch: cublasSgemmStridedBatched.
        // We'll use cublasSgemmStridedBatched to compute scores = Q_h * K_h^T / sqrt(head_dim)
        // Q_h: [batch, seq, head_dim] but stored with stride embed_dim between tokens and head_dim offset.
        // To use GEMM, we need to extract Q_h into a contiguous buffer? Or we can use strided batched GEMM.
        // cublasSgemmStridedBatched allows specifying stride between batches.
        // For simplicity, we'll implement a custom kernel for attention, but that might be slower than optimized GEMM.
        // Since performance is important, we should use cuBLAS for these GEMMs.
        // We'll create temporary contiguous buffers for Q_h, K_h, V_h for each head.
        // That would require copying data for each head, but total data size is batch*seq*head_dim, manageable.
        // We'll allocate q_head, k_head, v_head of size batch*seq*head_dim.
        float *d_q_head, *d_k_head, *d_v_head;
        CUDA_CHECK(cudaMalloc(&d_q_head, batch * seq * head_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_k_head, batch * seq * head_dim * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_v_head, batch * seq * head_dim * sizeof(float)));
        // Copy from d_q, d_k, d_v by extracting the slice for this head.
        // Launch copy kernel: for each element in q_head, map to corresponding in d_q with offset h*head_dim.
        copy_head_kernel<<<(batch*seq*head_dim + 255)/256, 256, 0, stream>>>(
            d_q_head, d_q, batch*seq, embed_dim, head_dim, h,
            d_k_head, d_k, batch*seq, embed_dim, head_dim, h,
            d_v_head, d_v, batch*seq, embed_dim, head_dim, h);

        // Compute scores = Q_h * K_h^T / sqrt(head_dim) using strided batched GEMM.
        // Each batch is independent: C_b = A_b * B_b^T, where A_b is [seq, head_dim], B_b is [seq, head_dim], C_b is [seq, seq].
        // We can use cublasSgemmStridedBatched with batchCount = batch.
        // Parameters: handle, transA, transB, m=seq, n=seq, k=head_dim,
        // alpha, A, lda=seq, strideA=seq*head_dim, B, ldb=seq, strideB=seq*head_dim, beta, C, ldc=seq, strideC=seq*seq, batchCount=batch
        float scale = 1.0f / sqrtf((float)head_dim);
        beta = 0.0f;
        status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                           seq, seq, head_dim,
                                           &scale, d_q_head, seq, seq*head_dim,
                                           d_k_head, seq, seq*head_dim,
                                           &beta, d_scores, seq, seq*seq,
                                           batch);
        if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "Scores GEMM failed\n"); return; }

        // Softmax over last dimension (each row of each batch)
        softmax_kernel<<<batch*seq, 256, 0, stream>>>(d_scores, d_probs, seq);

        // Accumulate weights (average over heads): d_weights_sum[b, i, j] += d_probs[b, i, j] / num_heads
        accumulate_weights_kernel<<<(batch*seq*seq + 255)/256, 256, 0, stream>>>(
            d_weights_sum, d_probs, batch, seq, 1.0f / num_heads);

        // Compute attn_out_h = probs * V_h [batch, seq, head_dim]
        // Use cublasSgemmStridedBatched: C_b = A_b * B_b, where A_b = probs [seq, seq], B_b = V_h [seq, head_dim]
        // A stored as [batch, seq, seq] with stride seq*seq
        // B stored as [batch, seq, head_dim] with stride seq*head_dim
        // C stored in d_attn_before_out at offset h*head_dim for each token.
        // We'll allocate a temp output for this head: d_attn_head [batch, seq, head_dim]
        float *d_attn_head;
        CUDA_CHECK(cudaMalloc(&d_attn_head, batch * seq * head_dim * sizeof(float)));
        alpha = 1.0f;
        beta = 0.0f;
        status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                           seq, head_dim, seq,
                                           &alpha, d_probs, seq, seq*seq,
                                           d_v_head, seq, seq*head_dim,
                                           &beta, d_attn_head, seq, seq*head_dim,
                                           batch);
        if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "Attn GEMM failed\n"); return; }

        // Copy d_attn_head back into d_attn_before_out with interleaving: attn_before_out[t*embed_dim + h*head_dim + i] = attn_head[b, s, i]
        // We need to map from head-contiguous to interleaved.
        copy_back_interleaved_kernel<<<(batch*seq*head_dim + 255)/256, 256, 0, stream>>>(
            d_attn_before_out, d_attn_head, batch*seq, embed_dim, head_dim, h);

        // Free head buffers
        cudaFree(d_q_head);
        cudaFree(d_k_head);
        cudaFree(d_v_head);
        cudaFree(d_attn_head);
    }

    // Step 3: Out projection: attn_output = attn_before_out * out_proj_weight^T + out_proj_bias
    // attn_before_out [batch*seq, embed_dim], out_proj_weight [embed_dim, embed_dim]
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                         total_seq, embed_dim, embed_dim,
                         &alpha, d_attn_before_out, total_seq, out_proj_weight, embed_dim,
                         &beta, attn_output, total_seq);
    if (status != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "GEMM out_proj failed\n"); return; }
    // Add out_proj_bias
    {
        dim3 block(256);
        dim3 grid((total_seq * embed_dim + block.x - 1) / block.x);
        add_bias_kernel<<<grid, block, 0, stream>>>(attn_output, out_proj_bias, total_seq, embed_dim);
    }

    // Copy d_weights_sum to attn_output_weights (already average)
    CUDA_CHECK(cudaMemcpyAsync(attn_output_weights, d_weights_sum, batch * seq * seq * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // Cleanup
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_weights_sum);
    cudaFree(d_scores);
    cudaFree(d_probs);
    cudaFree(d_attn_before_out);
    cublasDestroy(handle);
}

// ========================== Helper Kernels ==========================

__global__ void add_bias_kernel(float* __restrict__ data, const float* __restrict__ bias, int64_t rows, int64_t cols) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = rows * cols;
    if (idx < total) {
        int64_t col = idx % cols;
        int64_t row = idx / cols;
        data[idx] += bias[col];
    }
}

__global__ void copy_head_kernel(
    float* __restrict__ q_head, const float* __restrict__ q_full,
    float* __restrict__ k_head, const float* __restrict__ k_full,
    float* __restrict__ v_head, const float* __restrict__ v_full,
    int64_t total_tokens, int64_t embed_dim, int64_t head_dim, int64_t head_idx)
{
    // total elements per head = total_tokens * head_dim
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = total_tokens * head_dim;
    if (idx < total) {
        int64_t token = idx / head_dim;
        int64_t d = idx % head_dim;
        q_head[idx] = q_full[token * embed_dim + head_idx * head_dim + d];
        k_head[idx] = k_full[token * embed_dim + head_idx * head_dim + d];
        v_head[idx] = v_full[token * embed_dim + head_idx * head_dim + d];
    }
}

__global__ void copy_back_interleaved_kernel(
    float* __restrict__ interleaved, const float* __restrict__ head_data,
    int64_t total_tokens, int64_t embed_dim, int64_t head_dim, int64_t head_idx)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = total_tokens * head_dim;
    if (idx < total) {
        int64_t token = idx / head_dim;
        int64_t d = idx % head_dim;
        interleaved[token * embed_dim + head_idx * head_dim + d] = head_data[idx];
    }
}

__global__ void softmax_kernel(float* __restrict__ scores, float* __restrict__ probs, int64_t seq_len) {
    // Each block handles one row (one batch b, one query position i)
    // Grid dimension: batch * seq blocks
    int64_t batch_idx = blockIdx.x / seq_len; // which batch
    int64_t row_idx = blockIdx.x % seq_len;   // which query position
    float* row = scores + (batch_idx * seq_len + row_idx) * seq_len;
    float* out_row = probs + (batch_idx * seq_len + row_idx) * seq_len;
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int64_t j = 0; j < seq_len; ++j) {
        max_val = fmaxf(max_val, row[j]);
    }
    // Compute exp and sum
    float sum = 0.0f;
    for (int64_t j = 0; j < seq_len; ++j) {
        float e = expf(row[j] - max_val);
        out_row[j] = e;
        sum += e;
    }
    // Normalize
    float inv_sum = 1.0f / sum;
    for (int64_t j = 0; j < seq_len; ++j) {
        out_row[j] *= inv_sum;
    }
}

__global__ void accumulate_weights_kernel(
    float* __restrict__ weights_sum, const float* __restrict__ probs,
    int64_t batch, int64_t seq_len, float scale)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = batch * seq_len * seq_len;
    if (idx < total) {
        weights_sum[idx] += probs[idx] * scale;
    }
}

// log10_abs_eps launcher
// Parameters: x pointer, out pointer, numel, eps
// Assumes contiguous x and out with same shape.
extern "C" void log10_abs_eps_launcher(
    const float* x, float* out, long long numel, double eps, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    // Cap blocks to a reasonable number, but since kernel has grid-stride loop, we can use a smaller grid.
    // We'll use a fixed maximum grid size to avoid launching too many blocks.
    int max_blocks = 65535; // arbitrary
    if (blocks > max_blocks) blocks = max_blocks;
    log10_abs_eps_kernel<<<blocks, threads, 0, stream>>>(x, out, numel, (float)eps);
}
