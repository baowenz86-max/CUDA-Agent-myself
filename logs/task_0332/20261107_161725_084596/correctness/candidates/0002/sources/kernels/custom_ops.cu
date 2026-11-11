// CUDA kernels and launchers for custom operations
// Only CUDA kernels and launchers with raw pointers/scalars/cudaStream_t.
// No torch/ATen/c10/pybind dependencies.

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <algorithm>

// ------------------------- Helper functions -------------------------

__device__ inline float atomicMaxFloat(float* address, float val) {
    int* address_as_i = reinterpret_cast<int*>(address);
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// ------------------------- AdaptiveAvgPool1d -------------------------
// Input: x shape [batch, channels, seq_len], arbitrary strides in last two dims.
// Output: y shape [batch, channels, output_size], contiguous (stride output_size, 1).
// Computes adaptive average pooling: for each output index i, start = floor(i * seq_len / output_size),
// end = ceil((i+1) * seq_len / output_size), average over [start, end).

__global__ void adaptive_avg_pool1d_kernel(const float* __restrict__ x,
                                           float* __restrict__ y,
                                           int batch,
                                           int channels,
                                           int input_length,
                                           int output_length,
                                           int in_stride_batch,
                                           int in_stride_channel,
                                           int in_stride_length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * output_length;
    if (idx >= total) return;

    int out_idx = idx;
    int out_l = out_idx % output_length;
    int out_c = (out_idx / output_length) % channels;
    int out_b = out_idx / (output_length * channels);

    int start = (int)floorf((float)out_l * input_length / output_length);
    int end = (int)ceilf((float)(out_l + 1) * input_length / output_length);
    if (end > input_length) end = input_length;
    int count = end - start;
    if (count <= 0) {
        y[idx] = 0.0f;
        return;
    }

    const float* x_base = x + (size_t)out_b * in_stride_batch + (size_t)out_c * in_stride_channel;
    float sum = 0.0f;
    for (int i = start; i < end; ++i) {
        sum += x_base[(size_t)i * in_stride_length];
    }
    y[idx] = sum / count;
}

// Launcher
// x_data: raw float pointer to input
// batch, channels, input_length, output_length
// input strides (batch, channel, length) in elements
// output assumed contiguous [batch, channels, output_length]
extern "C" void adaptive_avg_pool1d_launcher(const float* x_data,
                                              float* y_data,
                                              int batch,
                                              int channels,
                                              int input_length,
                                              int output_length,
                                              int in_stride_batch,
                                              int in_stride_channel,
                                              int in_stride_length,
                                              cudaStream_t stream) {
    int total = batch * channels * output_length;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    adaptive_avg_pool1d_kernel<<<blocks, threads, 0, stream>>>(
        x_data, y_data, batch, channels, input_length, output_length,
        in_stride_batch, in_stride_channel, in_stride_length);
}

// ------------------------- Transpose (copy to contiguous) -------------------------
// Input: x shape [batch, dim0_size, dim1_size] (or any 3D with dims to swap)
// We assume exactly 3D and swap dims 1 and 2, but the launcher is general for swapping dim0 and dim1.
// Output: y shape [batch, dim1_size, dim0_size], contiguous.
__global__ void transpose_copy_kernel(const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int batch,
                                       int dim0_size,
                                       int dim1_size,
                                       int in_stride_batch,
                                       int in_stride_dim0,
                                       int in_stride_dim1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dim0_size * dim1_size;
    if (idx >= total) return;

    int i1 = idx % dim1_size;       // original dim1 index
    int i0 = (idx / dim1_size) % dim0_size; // original dim0 index
    int b = idx / (dim1_size * dim0_size);

    const float* src = x + (size_t)b * in_stride_batch + (size_t)i0 * in_stride_dim0 + (size_t)i1 * in_stride_dim1;
    // Output is contiguous: y[b][i1][i0] = y[b * dim1_size * dim0_size + i1 * dim0_size + i0]
    int out_idx = b * dim1_size * dim0_size + i1 * dim0_size + i0;
    y[out_idx] = *src;
}

extern "C" void transpose_copy_launcher(const float* x_data,
                                         float* y_data,
                                         int batch,
                                         int dim0_size,
                                         int dim1_size,
                                         int in_stride_batch,
                                         int in_stride_dim0,
                                         int in_stride_dim1,
                                         cudaStream_t stream) {
    int total = batch * dim0_size * dim1_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    transpose_copy_kernel<<<blocks, threads, 0, stream>>>(
        x_data, y_data, batch, dim0_size, dim1_size,
        in_stride_batch, in_stride_dim0, in_stride_dim1);
}

// ------------------------- Flatten -------------------------
// Input x shape [batch, seq_len, hidden_size] with arbitrary strides.
// Output y contiguous shape [batch, seq_len*hidden_size].
// If input already contiguous, copy; otherwise strided copy.

__global__ void flatten_copy_kernel(const float* __restrict__ x,
                                     float* __restrict__ y,
                                     int batch,
                                     int seq_len,
                                     int hidden_size,
                                     int in_stride_batch,
                                     int in_stride_seq,
                                     int in_stride_hidden) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq_len * hidden_size;
    if (idx >= total) return;

    int h = idx % hidden_size;
    int s = (idx / hidden_size) % seq_len;
    int b = idx / (hidden_size * seq_len);

    const float* src = x + (size_t)b * in_stride_batch + (size_t)s * in_stride_seq + (size_t)h * in_stride_hidden;
    y[idx] = *src;
}

extern "C" void flatten_launcher(const float* x_data,
                                  float* y_data,
                                  int batch,
                                  int seq_len,
                                  int hidden_size,
                                  int in_stride_batch,
                                  int in_stride_seq,
                                  int in_stride_hidden,
                                  cudaStream_t stream) {
    int total = batch * seq_len * hidden_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    flatten_copy_kernel<<<blocks, threads, 0, stream>>>(
        x_data, y_data, batch, seq_len, hidden_size,
        in_stride_batch, in_stride_seq, in_stride_hidden);
}

// ------------------------- Linear -------------------------
// y = x @ W^T + b, x [M,K], W [N,K], b [N], y [M,N]
// Simple GEMM kernel, one thread per output element.

__global__ void linear_fp32_kernel(const float* __restrict__ x,
                                   const float* __restrict__ w,
                                   const float* __restrict__ b,
                                   float* __restrict__ y,
                                   int M, int K, int N,
                                   bool has_bias) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += x[row * K + k] * w[col * K + k];
    }
    if (has_bias) sum += b[col];
    y[row * N + col] = sum;
}

extern "C" void linear_launcher(const float* x_data,
                                 const float* w_data,
                                 const float* b_data,
                                 float* y_data,
                                 int batch_size,
                                 int in_features,
                                 int out_features,
                                 int has_bias,
                                 cudaStream_t stream) {
    dim3 threads(256);
    dim3 blocks((out_features + threads.x - 1) / threads.x, batch_size);
    linear_fp32_kernel<<<blocks, threads, 0, stream>>>(
        x_data, w_data, b_data, y_data,
        batch_size, in_features, out_features, has_bias != 0);
}

// ------------------------- LSTM placeholder launcher -------------------------
// The actual LSTM will be handled by C++ binding using cuBLAS/cuDNN.
// This launcher is declared for interface consistency but not used.
extern "C" void lstm_launcher(const float* input,
                               float* output,
                               const float* weight_ih,
                               const float* weight_hh,
                               const float* bias_ih,
                               const float* bias_hh,
                               int batch_size,
                               int seq_len,
                               int input_size,
                               int hidden_size,
                               int num_layers,
                               int batch_first,
                               cudaStream_t stream) {
    // Not implemented in .cu; will be handled in C++ with cuBLAS.
    // This function is intentionally left empty to satisfy linker.
}

// ------------------------- End of file -------------------------
