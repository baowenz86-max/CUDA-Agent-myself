#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdint>

// Helper to check CUDA errors
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,       \
              (int)status);                                                    \
      return;                                                                  \
    }                                                                          \
  } while (0)

// Softmax kernel along the last dimension, supporting arbitrary strides.
// grid = (batch * seq_len), block = (block_size)
// For each row (B*S rows), compute max, then exp-sum, then normalize.
// We assume dim_size = embed_dim = 64, which is small enough for one block.
__global__ void softmax_last_dim_kernel(const float* __restrict__ input,
                                        float* __restrict__ output,
                                        int64_t batch, int64_t seq_len,
                                        int64_t dim_size,
                                        int64_t in_batch_stride,
                                        int64_t in_seq_stride,
                                        int64_t in_dim_stride,
                                        int64_t out_batch_stride,
                                        int64_t out_seq_stride,
                                        int64_t out_dim_stride) {
  // Each block handles one row: batch b, seq s
  int64_t row = blockIdx.x;
  if (row >= batch * seq_len) return;
  int64_t b = row / seq_len;
  int64_t s = row % seq_len;

  const float* in_row = input + b * in_batch_stride + s * in_seq_stride;
  float* out_row = output + b * out_batch_stride + s * out_seq_stride;

  // Compute max
  float max_val = -INFINITY;
  for (int64_t d = 0; d < dim_size; ++d) {
    float v = in_row[d * in_dim_stride];
    if (v > max_val) max_val = v;
  }

  // Compute sum of exp
  float sum = 0.0f;
  for (int64_t d = 0; d < dim_size; ++d) {
    float v = in_row[d * in_dim_stride];
    sum += expf(v - max_val);
  }

  // Normalize and write output
  for (int64_t d = 0; d < dim_size; ++d) {
    float v = in_row[d * in_dim_stride];
    out_row[d * out_dim_stride] = expf(v - max_val) / sum;
  }
}

// Launcher for custom_softmax
// input/output are assumed to be float32 and on CUDA.
// dim must be -1 (last dimension), but we check.
extern "C" void custom_softmax_launcher(
    const float* input, float* output,
    int64_t batch, int64_t seq_len, int64_t dim_size,
    int64_t in_batch_stride, int64_t in_seq_stride, int64_t in_dim_stride,
    int64_t out_batch_stride, int64_t out_seq_stride, int64_t out_dim_stride,
    int64_t dim, cudaStream_t stream) {
  if (dim != -1 && dim != 2) {
    // Only support last dimension per contract
    return;
  }
  int64_t rows = batch * seq_len;
  if (rows == 0 || dim_size == 0) return;
  // Use 256 threads per block, dim_size typically 64, so one thread per element with loops.
  int threads = 256;
  int blocks = (int)rows;
  softmax_last_dim_kernel<<<blocks, threads, 0, stream>>>(
      input, output, batch, seq_len, dim_size,
      in_batch_stride, in_seq_stride, in_dim_stride,
      out_batch_stride, out_seq_stride, out_dim_stride);
  CUDA_CHECK(cudaGetLastError());
}

// Attention: out = softmax(Q * K^T / sqrt(dim)) * V
// Q,K,V shape [batch, seq_len, dim]
// We treat as batch matrix multiplications:
//   scores = Q (B*S, D) * K^T (D, B*S)  ==> (B*S, B*S) per batch? Actually for each batch, S x S.
//   We can reshape to batch matrices: Q_b [S,D], K_b [S,D], V_b [S,D]
//   scores_b [S,S] = Q_b * K_b^T / sqrt(D)
//   softmax over last dim (rows)
//   out_b [S,D] = softmax_scores_b * V_b
// We'll use cuBLAS batched GEMM, or loop over batch.
// Given the small S and D, and B=512, we can use cublasSgemmStridedBatched for efficiency.
// However, to keep code simpler and robust, we can use cublasSgemm for each batch sequentially,
// but B=512 might be okay. More efficient: use strided batched GEMM.
// We'll implement with cublasSgemmStridedBatched.

// Softmax over rows of scores for each batch. scores shape [B, S, S] (or contiguous per batch)
// We'll compute softmax over last dim (S) for each row (b,s).
__global__ void softmax_scores_kernel(float* scores, int B, int S) {
  int row = blockIdx.x;
  int b = blockIdx.y;
  if (b >= B || row >= S) return;
  float* scores_b = scores + b * (int64_t)S * S;
  float* scores_row = scores_b + row * S;
  float max_val = -INFINITY;
  for (int i = 0; i < S; ++i) {
    if (scores_row[i] > max_val) max_val = scores_row[i];
  }
  float sum = 0.0f;
  for (int i = 0; i < S; ++i) {
    sum += expf(scores_row[i] - max_val);
  }
  for (int i = 0; i < S; ++i) {
    scores_row[i] = expf(scores_row[i] - max_val) / sum;
  }
}

// Launcher for custom_scaled_dot_product_attention
// Inputs may have arbitrary strides; we'll copy to contiguous temporaries for cuBLAS.
// Output stride is specified by out_batch_stride, out_seq_stride, out_dim_stride (expected [64, 32768, 1])
// We'll allocate temporary contiguous buffers for Q,K,V and output.
// The function signature includes all necessary strides.
extern "C" void custom_scaled_dot_product_attention_launcher(
    const float* query, const float* key, const float* value,
    float* output,
    int64_t batch, int64_t seq_len, int64_t dim_size,
    int64_t q_batch_stride, int64_t q_seq_stride, int64_t q_dim_stride,
    int64_t k_batch_stride, int64_t k_seq_stride, int64_t k_dim_stride,
    int64_t v_batch_stride, int64_t v_seq_stride, int64_t v_dim_stride,
    int64_t out_batch_stride, int64_t out_seq_stride, int64_t out_dim_stride,
    cudaStream_t stream) {
  if (batch == 0 || seq_len == 0 || dim_size == 0) return;
  // Create cuBLAS handle for this stream
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  CUBLAS_CHECK(cublasSetStream(handle, stream));

  // Temporary contiguous buffers for Q,K,V and output (will be transposed to output strides)
  float *Q_contig, *K_contig, *V_contig, *O_contig;
  int64_t total_batch_elems = batch * seq_len;
  int64_t size_qkv = total_batch_elems * dim_size;
  int64_t size_scores = batch * seq_len * seq_len; // B*S*S
  int64_t size_o_contig = total_batch_elems * dim_size;

  CUDA_CHECK(cudaMalloc(&Q_contig, size_qkv * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&K_contig, size_qkv * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&V_contig, size_qkv * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&O_contig, size_o_contig * sizeof(float)));
  float* scores;
  CUDA_CHECK(cudaMalloc(&scores, size_scores * sizeof(float)));

  // Copy Q,K,V into contiguous buffers respecting strides
  // We'll use cudaMemcpy2D for each tensor? Simpler: launch a kernel to copy with strides.
  // Or use cudaMemcpy for each batch row? For efficiency, write a copy kernel.
  __global__ void copy_tensor_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                     int64_t batch, int64_t seq_len, int64_t dim_size,
                                     int64_t src_batch_stride, int64_t src_seq_stride, int64_t src_dim_stride) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch * seq_len * dim_size;
    if (idx >= total) return;
    int64_t d = idx % dim_size;
    int64_t tmp = idx / dim_size;
    int64_t s = tmp % seq_len;
    int64_t b = tmp / seq_len;
    dst[idx] = src[b * src_batch_stride + s * src_seq_stride + d * src_dim_stride];
  }

  // Use grid-stride copy with 256 threads
  int copy_threads = 256;
  int copy_blocks = (size_qkv + copy_threads - 1) / copy_threads;
  // copy Q
  copy_tensor_kernel<<<copy_blocks, copy_threads, 0, stream>>>(
      query, Q_contig, batch, seq_len, dim_size, q_batch_stride, q_seq_stride, q_dim_stride);
  CUDA_CHECK(cudaGetLastError());
  // copy K
  copy_tensor_kernel<<<copy_blocks, copy_threads, 0, stream>>>(
      key, K_contig, batch, seq_len, dim_size, k_batch_stride, k_seq_stride, k_dim_stride);
  CUDA_CHECK(cudaGetLastError());
  // copy V
  copy_tensor_kernel<<<copy_blocks, copy_threads, 0, stream>>>(
      value, V_contig, batch, seq_len, dim_size, v_batch_stride, v_seq_stride, v_dim_stride);
  CUDA_CHECK(cudaGetLastError());

  // Compute scores = Q_contig * K_contig^T / sqrt(dim)
  // Q_contig is (B, S, D) contiguous, K_contig same.
  // For each batch, we need SxS matrix. We can use cublasSgemmStridedBatched.
  float alpha = 1.0f / sqrtf((float)dim_size);
  float beta = 0.0f;
  int m = (int)seq_len;
  int n = (int)seq_len;
  int k = (int)dim_size;
  int64_t strideA = (int64_t)seq_len * dim_size;
  int64_t strideB = (int64_t)seq_len * dim_size;
  int64_t strideC = (int64_t)seq_len * seq_len;
  CUBLAS_CHECK(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_T,
      m, n, k, &alpha,
      Q_contig, m, strideA,
      K_contig, n, strideB,
      &beta,
      scores, m, strideC,
      (int)batch));

  // Softmax over scores for each batch, each row (last dim)
  // scores layout: [B, S, S] contiguous, so for each batch, rows are S.
  // We can launch a 2D grid: x = row (S), y = batch (B).
  dim3 softmax_grid(m, (unsigned)batch);
  softmax_scores_kernel<<<softmax_grid, 1, 0, stream>>>(scores, (int)batch, m);
  CUDA_CHECK(cudaGetLastError());

  // Compute output = softmax_scores * V_contig
  // softmax_scores is [B, S, S] contiguous per batch, V_contig is [B, S, D] contiguous per batch.
  // For each batch: O_b = P_b * V_b, where P_b [S,S], V_b [S,D].
  // Use cublasSgemmStridedBatched with A = scores (B, S, S), B = V_contig (B, S, D), C = O_contig (B, S, D)
  float alpha2 = 1.0f;
  float beta2 = 0.0f;
  CUBLAS_CHECK(cublasSgemmStridedBatched(
      handle, CUBLAS_OP_N, CUBLAS_OP_N,
      m, (int)dim_size, m,
      &alpha2,
      scores, m, (int64_t)m * m,
      V_contig, (int)dim_size, (int64_t)m * dim_size,
      &beta2,
      O_contig, (int)dim_size, (int64_t)m * dim_size,
      (int)batch));

  // Copy O_contig to output with given strides
  // O_contig is [B,S,D] contiguous, output has out_batch_stride, out_seq_stride, out_dim_stride
  __global__ void copy_output_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                     int64_t batch, int64_t seq_len, int64_t dim_size,
                                     int64_t out_batch_stride, int64_t out_seq_stride, int64_t out_dim_stride) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch * seq_len * dim_size;
    if (idx >= total) return;
    int64_t d = idx % dim_size;
    int64_t tmp = idx / dim_size;
    int64_t s = tmp % seq_len;
    int64_t b = tmp / seq_len;
    dst[b * out_batch_stride + s * out_seq_stride + d * out_dim_stride] = src[idx];
  }

  int blocks_out = (size_o_contig + copy_threads - 1) / copy_threads;
  copy_output_kernel<<<blocks_out, copy_threads, 0, stream>>>(
      O_contig, output, batch, seq_len, dim_size, out_batch_stride, out_seq_stride, out_dim_stride);
  CUDA_CHECK(cudaGetLastError());

  // Cleanup
  cudaFree(Q_contig);
  cudaFree(K_contig);
  cudaFree(V_contig);
  cudaFree(scores);
  cudaFree(O_contig);
  cublasDestroy(handle);
}
