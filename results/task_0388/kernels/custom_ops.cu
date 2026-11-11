#include <cuda_runtime.h>
#include <cstdint>

// Helper: grid-stride loop
#define CUDA_KERNEL_LOOP(i, n) \
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// -----------------------------------------------------------------------------
// embedding_forward
//   input: int64_t, shape [batch_size, seq_length]
//   weight: float, shape [num_embeddings, embedding_dim]
//   output: float, shape [batch_size, seq_length, embedding_dim]
// -----------------------------------------------------------------------------
__global__ void embedding_forward_kernel(
    const int64_t* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t total_elements,
    int64_t seq_length,
    int64_t embedding_dim) {
  CUDA_KERNEL_LOOP(idx, total_elements) {
    int64_t batch = idx / (seq_length * embedding_dim);
    int64_t rem = idx % (seq_length * embedding_dim);
    int64_t seq = rem / embedding_dim;
    int64_t dim = rem % embedding_dim;
    int64_t input_idx = batch * seq_length + seq;
    int64_t token = input[input_idx];
    // Assume token is valid (< num_embeddings); binding.cpp should validate.
    output[idx] = weight[token * embedding_dim + dim];
  }
}

extern "C" void embedding_forward_launcher(
    const int64_t* input,
    const float* weight,
    float* output,
    int64_t batch_size,
    int64_t seq_length,
    int64_t embedding_dim,
    cudaStream_t stream) {
  int64_t total_elements = batch_size * seq_length * embedding_dim;
  if (total_elements <= 0) return;
  int threads = 256;
  int blocks = (total_elements + threads - 1) / threads;
  // Cap grid size to avoid excessive blocks if total_elements is huge.
  if (blocks > 65535) blocks = 65535;
  embedding_forward_kernel<<<blocks, threads, 0, stream>>>(
      input, weight, output, total_elements, seq_length, embedding_dim);
}

// -----------------------------------------------------------------------------
// mean_dim_forward
//   input: float, shape [batch_size, seq_length, embedding_dim]
//   output: float, shape [batch_size, embedding_dim]
//   dim is fixed to 1, but passed as int64_t for interface compatibility.
// -----------------------------------------------------------------------------
__global__ void mean_dim_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t seq_length,
    int64_t embedding_dim) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total_outputs = batch_size * embedding_dim;
  if (idx >= total_outputs) return;
  int64_t batch = idx / embedding_dim;
  int64_t dim = idx % embedding_dim;
  float sum = 0.0f;
  const float* input_ptr = input + batch * seq_length * embedding_dim + dim;
  for (int64_t s = 0; s < seq_length; ++s) {
    sum += input_ptr[s * embedding_dim];
  }
  output[idx] = sum / static_cast<float>(seq_length);
}

extern "C" void mean_dim_forward_launcher(
    const float* input,
    float* output,
    int64_t batch_size,
    int64_t seq_length,
    int64_t embedding_dim,
    int64_t dim,
    cudaStream_t stream) {
  // Only dim=1 is supported by contract; but kernel ignores dim since input shape is fixed.
  if (dim != 1) return; // binding should handle error; silently return for safety.
  int64_t total_outputs = batch_size * embedding_dim;
  if (total_outputs <= 0) return;
  int threads = 256;
  int blocks = (total_outputs + threads - 1) / threads;
  mean_dim_forward_kernel<<<blocks, threads, 0, stream>>>(
      input, output, batch_size, seq_length, embedding_dim);
}

// -----------------------------------------------------------------------------
// linear_forward
//   input: float, shape [batch_size, in_features]
//   weight: float, shape [out_features, in_features]
//   bias: float, shape [out_features]
//   output: float, shape [batch_size, out_features]
// -----------------------------------------------------------------------------
__global__ void linear_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total_outputs = batch_size * out_features;
  if (idx >= total_outputs) return;
  int64_t batch = idx / out_features;
  int64_t out_idx = idx % out_features;
  const float* input_row = input + batch * in_features;
  float sum = 0.0f;
  for (int64_t k = 0; k < in_features; ++k) {
    sum += input_row[k] * weight[out_idx * in_features + k];
  }
  if (bias != nullptr) {
    sum += bias[out_idx];
  }
  output[idx] = sum;
}

extern "C" void linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    cudaStream_t stream) {
  int64_t total_outputs = batch_size * out_features;
  if (total_outputs <= 0) return;
  int threads = 256;
  int blocks = (total_outputs + threads - 1) / threads;
  linear_forward_kernel<<<blocks, threads, 0, stream>>>(
      input, weight, bias, output, batch_size, in_features, out_features);
}

// -----------------------------------------------------------------------------
// bce_with_logits_forward
//   logits: float, shape [batch_size, 1]
//   target: float, shape [batch_size, 1]
//   output: float scalar (shape []), mean loss
// -----------------------------------------------------------------------------
__global__ void bce_with_logits_sum_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ target,
    float* __restrict__ partial_sums,
    int64_t batch_size,
    int64_t block_size) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  float val = 0.0f;
  // grid-stride loop over batch elements
  for (int64_t j = i; j < batch_size; j += blockDim.x * gridDim.x) {
    float x = logits[j];
    float y = target[j];
    // stable BCE with logits: max(x,0) - x*y + log(1+exp(-|x|))
    float abs_x = fabsf(x);
    float loss = fmaxf(x, 0.0f) - x * y + log1pf(expf(-abs_x));
    val += loss;
  }
  sdata[tid] = val;
  __syncthreads();
  // block reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    partial_sums[blockIdx.x] = sdata[0];
  }
}

__global__ void bce_with_logits_finalize_kernel(
    const float* __restrict__ partial_sums,
    float* __restrict__ output,
    int64_t num_blocks,
    int64_t batch_size) {
  // single thread finalizes sum and divides by batch_size
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  float total = 0.0f;
  for (int64_t i = 0; i < num_blocks; ++i) {
    total += partial_sums[i];
  }
  output[0] = total / static_cast<float>(batch_size);
}

extern "C" void bce_with_logits_forward_launcher(
    const float* logits,
    const float* target,
    float* output,
    int64_t batch_size,
    cudaStream_t stream) {
  if (batch_size <= 0) return;
  int threads = 256;
  int blocks = (batch_size + threads - 1) / threads;
  // Cap blocks to avoid excessive partial_sums memory; we'll use grid-stride.
  int max_blocks = 1024; // arbitrary cap
  if (blocks > max_blocks) blocks = max_blocks;
  // Allocate temp partial sums on device (use cudaMallocAsync? simple cudaMalloc works)
  float* partial_sums = nullptr;
  cudaError_t err = cudaMalloc(&partial_sums, blocks * sizeof(float));
  if (err != cudaSuccess) {
    // in case of error, perhaps set output to NaN? Better to let binding handle.
    return;
  }
  int shared_mem_size = threads * sizeof(float);
  bce_with_logits_sum_kernel<<<blocks, threads, shared_mem_size, stream>>>(
      logits, target, partial_sums, batch_size, threads);
  bce_with_logits_finalize_kernel<<<1, 1, 0, stream>>>(
      partial_sums, output, blocks, batch_size);
  cudaFree(partial_sums);
}
