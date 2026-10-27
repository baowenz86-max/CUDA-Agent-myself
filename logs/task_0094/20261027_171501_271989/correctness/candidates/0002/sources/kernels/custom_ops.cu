// custom_ops.cu
// CUDA kernels and launchers for helper operations used in conv_transpose1d and flatten
// No PyTorch dependencies, only raw pointers, scalars, and cudaStream_t

#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Flatten kernel: copies input elements to output with same layout
// Input: [batch, channels, length], output: [batch, channels * length]
// Since both are contiguous, a 1D elementwise copy suffices.
// -----------------------------------------------------------------------------
__global__ void flatten_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               long long total_elements) {
  long long idx = blockIdx.x * blockDim.x + threadIdx.x;
  long long stride = blockDim.x * gridDim.x;
  for (long long i = idx; i < total_elements; i += stride) {
    output[i] = input[i];
  }
}

// -----------------------------------------------------------------------------
// Weight transpose kernel: converts PyTorch weight layout [in_channels, out_channels, kernel_size]
// to cuDNN layout [out_channels, in_channels, kernel_size]
// -----------------------------------------------------------------------------
__global__ void weight_transpose_kernel(const float* __restrict__ src,
                                        float* __restrict__ dst,
                                        int in_channels,
                                        int out_channels,
                                        int kernel_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = in_channels * out_channels * kernel_size;
  if (idx < total) {
    int k = idx % kernel_size;
    int oc = (idx / kernel_size) % out_channels;
    int ic = idx / (kernel_size * out_channels);
    int src_idx = (ic * out_channels + oc) * kernel_size + k;
    int dst_idx = (oc * in_channels + ic) * kernel_size + k;
    dst[dst_idx] = src[src_idx];
  }
}

// -----------------------------------------------------------------------------
// Copy and zero pad kernel: copies data from a shorter (no output_padding) buffer
// to the full output buffer, zeroing the extra columns at the end of each row.
// src has L_out_no_pad elements per (batch, channel) row,
// dst has L_out elements per row.
// -----------------------------------------------------------------------------
__global__ void copy_and_zero_pad_kernel(const float* __restrict__ src,
                                         float* __restrict__ dst,
                                         int total_rows,
                                         int L_out_no_pad,
                                         int L_out) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < total_rows) {
    const float* src_row = src + (long long)row * L_out_no_pad;
    float* dst_row = dst + (long long)row * L_out;
    for (int i = 0; i < L_out_no_pad; ++i) {
      dst_row[i] = src_row[i];
    }
    for (int i = L_out_no_pad; i < L_out; ++i) {
      dst_row[i] = 0.0f;
    }
  }
}

// -----------------------------------------------------------------------------
// Add bias kernel: adds bias[c] to each element in channel c.
// data shape: [batch, out_channels, L_out], bias shape: [out_channels]
// -----------------------------------------------------------------------------
__global__ void add_bias_kernel(float* data,
                                const float* bias,
                                int batch,
                                int out_channels,
                                int L_out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * out_channels * L_out;
  if (idx < total) {
    int c = (idx / L_out) % out_channels;
    data[idx] += bias[c];
  }
}

// -----------------------------------------------------------------------------
// Launchers (all return cudaError_t)
// -----------------------------------------------------------------------------

cudaError_t flatten_launcher(const float* input,
                             float* output,
                             long long total_elements,
                             cudaStream_t stream) {
  if (total_elements <= 0) {
    return cudaSuccess;
  }
  const int block_size = 256;
  long long blocks = (total_elements + block_size - 1) / block_size;
  if (blocks > 65535) {
    blocks = 65535;
  }
  flatten_kernel<<<(int)blocks, block_size, 0, stream>>>(input, output, total_elements);
  return cudaGetLastError();
}

cudaError_t weight_transpose_launcher(const float* src,
                                      float* dst,
                                      int in_channels,
                                      int out_channels,
                                      int kernel_size,
                                      cudaStream_t stream) {
  int total = in_channels * out_channels * kernel_size;
  if (total <= 0) {
    return cudaSuccess;
  }
  const int block_size = 256;
  int blocks = (total + block_size - 1) / block_size;
  weight_transpose_kernel<<<blocks, block_size, 0, stream>>>(src, dst,
                                                             in_channels, out_channels, kernel_size);
  return cudaGetLastError();
}

cudaError_t copy_and_zero_pad_launcher(const float* src,
                                       float* dst,
                                       int total_rows,
                                       int L_out_no_pad,
                                       int L_out,
                                       cudaStream_t stream) {
  if (total_rows <= 0) {
    return cudaSuccess;
  }
  const int block_size = 256;
  int blocks = (total_rows + block_size - 1) / block_size;
  copy_and_zero_pad_kernel<<<blocks, block_size, 0, stream>>>(src, dst,
                                                              total_rows, L_out_no_pad, L_out);
  return cudaGetLastError();
}

cudaError_t add_bias_launcher(float* data,
                              const float* bias,
                              int batch,
                              int out_channels,
                              int L_out,
                              cudaStream_t stream) {
  long long total = (long long)batch * out_channels * L_out;
  if (total <= 0) {
    return cudaSuccess;
  }
  const int block_size = 256;
  long long blocks = (total + block_size - 1) / block_size;
  if (blocks > 65535) {
    blocks = 65535;
  }
  add_bias_kernel<<<(int)blocks, block_size, 0, stream>>>(data, bias,
                                                          batch, out_channels, L_out);
  return cudaGetLastError();
}

// Extern "C" declarations for linking with .cpp bindings
extern "C" {
    cudaError_t flatten_launcher(const float* input, float* output, long long total_elements, cudaStream_t stream);
    cudaError_t weight_transpose_launcher(const float* src, float* dst, int in_channels, int out_channels, int kernel_size, cudaStream_t stream);
    cudaError_t copy_and_zero_pad_launcher(const float* src, float* dst, int total_rows, int L_out_no_pad, int L_out, cudaStream_t stream);
    cudaError_t add_bias_launcher(float* data, const float* bias, int batch, int out_channels, int L_out, cudaStream_t stream);
}
