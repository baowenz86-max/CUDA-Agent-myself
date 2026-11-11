#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// CUDA error check helper
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = call;                                                  \
    if (err != cudaSuccess) {                                                \
      printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
             cudaGetErrorString(err));                                       \
      return;                                                                \
    }                                                                        \
  } while (0)

// -----------------------------------------------------------------------------
// AdaptiveAvgPool1d kernel
// Input: x [batch, channels, input_length], output_size int
// Output: y [batch, channels, output_size]
// Each output element averages a contiguous segment of the input along last dim
// -----------------------------------------------------------------------------
__global__ void adaptive_avg_pool1d_kernel(const float* __restrict__ x,
                                           float* __restrict__ y,
                                           int batch,
                                           int channels,
                                           int input_length,
                                           int output_size) {
  int total = batch * channels * output_size;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < total; i += stride) {
    int oc = i % output_size;
    int tmp = i / output_size;
    int c = tmp % channels;
    int b = tmp / channels;

    int start = (int)((int64_t)oc * input_length / output_size);
    int end = (int)((int64_t)(oc + 1) * input_length / output_size);
    if (end <= start) end = start + 1;

    const float* in_ptr = x + ((int64_t)b * channels + c) * input_length;
    float sum = 0.0f;
    for (int k = start; k < end; ++k) {
      sum += in_ptr[k];
    }
    y[i] = sum / (float)(end - start);
  }
}

extern "C" void adaptive_avg_pool1d_launcher(const float* x,
                                             float* y,
                                             int batch,
                                             int channels,
                                             int input_length,
                                             int output_size,
                                             cudaStream_t stream) {
  int total = batch * channels * output_size;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  if (blocks > 1024) blocks = 1024;
  adaptive_avg_pool1d_kernel<<<blocks, threads, 0, stream>>>(
      x, y, batch, channels, input_length, output_size);
  CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// Transpose 1 <-> 2 kernel
// Input: x [batch, dim1, dim2], output y [batch, dim2, dim1]
// Handles arbitrary strides but assumes x is accessible via its strides
// -----------------------------------------------------------------------------
__global__ void transpose_1_2_kernel(const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int batch,
                                     int dim1,
                                     int dim2,
                                     int stride_batch_x,
                                     int stride_dim1_x,
                                     int stride_dim2_x) {
  int total = batch * dim1 * dim2;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < total; i += stride) {
    int d2 = i % dim2;
    int tmp = i / dim2;
    int d1 = tmp % dim1;
    int b = tmp / dim1;

    int src_idx = b * stride_batch_x + d1 * stride_dim1_x + d2 * stride_dim2_x;
    int dst_idx = (b * dim2 + d2) * dim1 + d1;
    y[dst_idx] = x[src_idx];
  }
}

extern "C" void transpose_1_2_launcher(const float* x,
                                       float* y,
                                       int batch,
                                       int dim1,
                                       int dim2,
                                       int stride_batch_x,
                                       int stride_dim1_x,
                                       int stride_dim2_x,
                                       cudaStream_t stream) {
  int total = batch * dim1 * dim2;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  if (blocks > 1024) blocks = 1024;
  transpose_1_2_kernel<<<blocks, threads, 0, stream>>>(
      x, y, batch, dim1, dim2, stride_batch_x, stride_dim1_x, stride_dim2_x);
  CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// Combine bias kernel: out = a + b
// -----------------------------------------------------------------------------
__global__ void combine_bias_kernel(float* out, const float* a, const float* b, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = a[idx] + b[idx];
}

// -----------------------------------------------------------------------------
// LSTM forward kernel (two layers, unidirectional)
// Input: x [batch, seq_len, input_size]
// Weights and biases for layer 0 and layer 1
// Output: y [batch, seq_len, hidden_size]
// Also writes h_n [2, batch, hidden_size], c_n [2, batch, hidden_size]
// -----------------------------------------------------------------------------
// Helper: sigmoid function
__device__ __forceinline__ float sigmoidf(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// Actual LSTM kernel: one block per batch, processes both layers.
__global__ void lstm_forward_combined_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight_ih_l0, // [4H, input_size]
    const float* __restrict__ weight_hh_l0, // [4H, H]
    const float* __restrict__ bias_l0,      // [4H]
    const float* __restrict__ weight_ih_l1, // [4H, H]
    const float* __restrict__ weight_hh_l1, // [4H, H]
    const float* __restrict__ bias_l1,      // [4H]
    float* __restrict__ y,                  // [batch, seq, H]
    float* __restrict__ h_n,                // [2, batch, H]
    float* __restrict__ c_n,                // [2, batch, H]
    int batch,
    int seq_len,
    int input_size,
    int hidden_size) {

  int bid = blockIdx.x;
  if (bid >= batch) return;

  const int H = hidden_size;
  const int I = input_size;

  // Shared memory: h0[H], c0[H], h1[H], c1[H]
  extern __shared__ float smem[];
  float* h0 = smem;
  float* c0 = smem + H;
  float* h1 = smem + 2 * H;
  float* c1 = smem + 3 * H;

  // Initialize hidden and cell states to zero
  int tid = threadIdx.x;
  int threads = blockDim.x;
  for (int i = tid; i < H; i += threads) {
    h0[i] = 0.0f;
    c0[i] = 0.0f;
    h1[i] = 0.0f;
    c1[i] = 0.0f;
  }
  __syncthreads();

  // Base pointers for this batch
  const float* x_b = x + (int64_t)bid * seq_len * I;
  float* y_b = y + (int64_t)bid * seq_len * H;

  // Gate arrays in shared memory: i0,f0,g0,o0,i1,f1,g1,o1
  float* i0 = smem + 4 * H;
  float* f0 = smem + 5 * H;
  float* g0 = smem + 6 * H;
  float* o0 = smem + 7 * H;
  float* i1 = smem + 8 * H;
  float* f1 = smem + 9 * H;
  float* g1 = smem + 10 * H;
  float* o1 = smem + 11 * H;

  for (int t = 0; t < seq_len; ++t) {
    // ----- Layer 0 -----
    const float* x_t = x_b + t * I;

    for (int j = tid; j < H; j += threads) {
      float pre_i = bias_l0[j];
      float pre_f = bias_l0[H + j];
      float pre_g = bias_l0[2 * H + j];
      float pre_o = bias_l0[3 * H + j];

      for (int k = 0; k < I; ++k) {
        float x_val = x_t[k];
        pre_i += weight_ih_l0[j * I + k] * x_val;
        pre_f += weight_ih_l0[(H + j) * I + k] * x_val;
        pre_g += weight_ih_l0[(2 * H + j) * I + k] * x_val;
        pre_o += weight_ih_l0[(3 * H + j) * I + k] * x_val;
      }
      for (int k = 0; k < H; ++k) {
        float h_val = h0[k];
        pre_i += weight_hh_l0[j * H + k] * h_val;
        pre_f += weight_hh_l0[(H + j) * H + k] * h_val;
        pre_g += weight_hh_l0[(2 * H + j) * H + k] * h_val;
        pre_o += weight_hh_l0[(3 * H + j) * H + k] * h_val;
      }

      i0[j] = sigmoidf(pre_i);
      f0[j] = sigmoidf(pre_f);
      g0[j] = tanhf(pre_g);
      o0[j] = sigmoidf(pre_o);
    }
    __syncthreads();

    for (int j = tid; j < H; j += threads) {
      c0[j] = f0[j] * c0[j] + i0[j] * g0[j];
      h0[j] = o0[j] * tanhf(c0[j]);
      y_b[t * H + j] = h0[j];
    }
    __syncthreads();

    // ----- Layer 1 -----
    const float* in1 = h0;

    for (int j = tid; j < H; j += threads) {
      float pre_i = bias_l1[j];
      float pre_f = bias_l1[H + j];
      float pre_g = bias_l1[2 * H + j];
      float pre_o = bias_l1[3 * H + j];

      for (int k = 0; k < H; ++k) {
        float val = in1[k];
        pre_i += weight_ih_l1[j * H + k] * val;
        pre_f += weight_ih_l1[(H + j) * H + k] * val;
        pre_g += weight_ih_l1[(2 * H + j) * H + k] * val;
        pre_o += weight_ih_l1[(3 * H + j) * H + k] * val;
      }
      for (int k = 0; k < H; ++k) {
        float h_val = h1[k];
        pre_i += weight_hh_l1[j * H + k] * h_val;
        pre_f += weight_hh_l1[(H + j) * H + k] * h_val;
        pre_g += weight_hh_l1[(2 * H + j) * H + k] * h_val;
        pre_o += weight_hh_l1[(3 * H + j) * H + k] * h_val;
      }

      i1[j] = sigmoidf(pre_i);
      f1[j] = sigmoidf(pre_f);
      g1[j] = tanhf(pre_g);
      o1[j] = sigmoidf(pre_o);
    }
    __syncthreads();

    for (int j = tid; j < H; j += threads) {
      c1[j] = f1[j] * c1[j] + i1[j] * g1[j];
      h1[j] = o1[j] * tanhf(c1[j]);
      y_b[t * H + j] = h1[j];
    }
    __syncthreads();
  }

  for (int j = tid; j < H; j += threads) {
    h_n[0 * batch * H + bid * H + j] = h0[j];
    c_n[0 * batch * H + bid * H + j] = c0[j];
    h_n[1 * batch * H + bid * H + j] = h1[j];
    c_n[1 * batch * H + bid * H + j] = c1[j];
  }
}

// Launcher for LSTM: combines biases, launches kernel with appropriate shared memory.
extern "C" void lstm_forward_launcher_actual(
    const float* x,
    const float* weight_ih_l0,
    const float* weight_hh_l0,
    const float* bias_ih_l0,
    const float* bias_hh_l0,
    const float* weight_ih_l1,
    const float* weight_hh_l1,
    const float* bias_ih_l1,
    const float* bias_hh_l1,
    float* y,
    float* h_n,
    float* c_n,
    int batch,
    int seq_len,
    int input_size,
    int hidden_size,
    cudaStream_t stream) {

  // Combine biases: bias = bias_ih + bias_hh
  float *bias_l0, *bias_l1;
  cudaMalloc(&bias_l0, 4 * hidden_size * sizeof(float));
  cudaMalloc(&bias_l1, 4 * hidden_size * sizeof(float));

  int total_bias = 4 * hidden_size;
  int threads_bias = 256;
  int blocks_bias = (total_bias + threads_bias - 1) / threads_bias;

  combine_bias_kernel<<<blocks_bias, threads_bias, 0, stream>>>(
      bias_l0, bias_ih_l0, bias_hh_l0, total_bias);
  combine_bias_kernel<<<blocks_bias, threads_bias, 0, stream>>>(
      bias_l1, bias_ih_l1, bias_hh_l1, total_bias);

  // Launch main LSTM kernel
  int threads = hidden_size; // assume hidden_size <= 1024, here 256
  if (threads > 1024) threads = 256; // fallback with loop
  // Shared memory: states (4*H) + gates (8*H) = 12*H floats
  size_t shmem = 12 * hidden_size * sizeof(float);
  lstm_forward_combined_kernel<<<batch, threads, shmem, stream>>>(
      x,
      weight_ih_l0, weight_hh_l0, bias_l0,
      weight_ih_l1, weight_hh_l1, bias_l1,
      y, h_n, c_n,
      batch, seq_len, input_size, hidden_size);

  CUDA_CHECK(cudaGetLastError());
  cudaFree(bias_l0);
  cudaFree(bias_l1);
}

// -----------------------------------------------------------------------------
// Reshape 2D kernel: copies data preserving order.
// Input x [batch, seq_len, hidden_size], output y [batch, seq_len*hidden_size]
// Handles arbitrary strides via explicit dimensions.
// -----------------------------------------------------------------------------
__global__ void reshape_2d_dim_kernel(const float* __restrict__ x,
                                      float* __restrict__ y,
                                      int batch,
                                      int seq_len,
                                      int hidden_size,
                                      int64_t stride_batch_x,
                                      int64_t stride_seq_x,
                                      int64_t stride_hidden_x) {
  int64_t total = (int64_t)batch * seq_len * hidden_size;
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t i = idx; i < total; i += stride) {
    int64_t h = i % hidden_size;
    int64_t tmp = i / hidden_size;
    int64_t s = tmp % seq_len;
    int64_t b = tmp / seq_len;
    int64_t src_idx = b * stride_batch_x + s * stride_seq_x + h * stride_hidden_x;
    y[i] = x[src_idx];
  }
}

extern "C" void reshape_2d_launcher_actual(const float* x,
                                           float* y,
                                           int batch,
                                           int seq_len,
                                           int hidden_size,
                                           int64_t stride_batch_x,
                                           int64_t stride_seq_x,
                                           int64_t stride_hidden_x,
                                           cudaStream_t stream) {
  int64_t total = (int64_t)batch * seq_len * hidden_size;
  int threads = 256;
  int64_t blocks64 = (total + threads - 1) / threads;
  int blocks = (blocks64 > 1024) ? 1024 : (int)blocks64;
  reshape_2d_dim_kernel<<<blocks, threads, 0, stream>>>(
      x, y, batch, seq_len, hidden_size,
      stride_batch_x, stride_seq_x, stride_hidden_x);
  CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// Linear kernel: y = x @ W^T + b
// x [batch, in_features], weight [out_features, in_features], bias [out_features]
// -----------------------------------------------------------------------------
__global__ void linear_kernel(const float* __restrict__ x,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ y,
                              int batch,
                              int in_features,
                              int out_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int total = batch * out_features;

  for (int i = idx; i < total; i += stride) {
    int out_idx = i % out_features;
    int b = i / out_features;
    const float* x_row = x + (int64_t)b * in_features;
    const float* w_row = weight + (int64_t)out_idx * in_features;
    float sum = (bias != nullptr) ? bias[out_idx] : 0.0f;
    for (int k = 0; k < in_features; ++k) {
      sum += x_row[k] * w_row[k];
    }
    y[i] = sum;
  }
}

extern "C" void linear_launcher(const float* x,
                                const float* weight,
                                const float* bias, // may be null
                                float* y,
                                int batch,
                                int in_features,
                                int out_features,
                                cudaStream_t stream) {
  int total = batch * out_features;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  if (blocks > 1024) blocks = 1024;
  linear_kernel<<<blocks, threads, 0, stream>>>(
      x, weight, bias, y, batch, in_features, out_features);
  CUDA_CHECK(cudaGetLastError());
}
