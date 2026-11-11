// CUDA kernels and launchers for custom operations
// Only CUDA kernels and launchers with raw pointers/scalars/cudaStream_t.
// No torch/ATen/c10/pybind dependencies.

#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <algorithm>

// ------------------------- Helper functions -------------------------

__device__ inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ inline float tanhf_approx(float x) {
    return tanhf(x);
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

// ------------------------- LSTM kernel -------------------------
// Implements a multi-layer LSTM without cuDNN, using raw CUDA kernels.
// Supports batch_first input, unidirectional LSTM, with optional bias.
// Input x shape [batch, seq_len, input_size]
// Weights: weight_ih_l{k} shape [4*hidden_size, input_size]; weight_hh_l{k} shape [4*hidden_size, hidden_size]
// biases: bias_ih_l{k}, bias_hh_l{k} shape [4*hidden_size]
// Output y shape [batch, seq_len, hidden_size]
// h_n, c_n shape [num_layers, batch, hidden_size]
// Hidden state initialization zeros.

__global__ void lstm_kernel(
    const float* __restrict__ x,          // [batch, seq_len, input_size]
    float* __restrict__ y,                // [batch, seq_len, hidden_size]
    float* __restrict__ h_n,              // [num_layers, batch, hidden_size]
    float* __restrict__ c_n,              // [num_layers, batch, hidden_size]
    const float* __restrict__ weight_ih_l0, // [4*hidden_size, input_size]
    const float* __restrict__ weight_hh_l0, // [4*hidden_size, hidden_size]
    const float* __restrict__ bias_ih_l0,   // [4*hidden_size]
    const float* __restrict__ bias_hh_l0,   // [4*hidden_size]
    const float* __restrict__ weight_ih_l1, // [4*hidden_size, hidden_size]
    const float* __restrict__ weight_hh_l1, // [4*hidden_size, hidden_size]
    const float* __restrict__ bias_ih_l1,   // [4*hidden_size]
    const float* __restrict__ bias_hh_l1,   // [4*hidden_size]
    int batch_size,
    int seq_len,
    int input_size,
    int hidden_size,
    int num_layers
) {
    // Each block handles one (batch, time) combination and computes all hidden units.
    int b = blockIdx.x / seq_len;
    int t = blockIdx.x % seq_len;
    int tid = threadIdx.x;

    extern __shared__ float smem[];
    // Shared memory layout:
    // For each layer, we need input to layer (prev_h or x_t) and calculate gates.
    // We'll use a direct per-thread loop over hidden units, each thread computes a subset of hidden units.
    // This is not highly optimized but correct.
    // We'll have each thread compute one hidden unit for each layer sequentially.
    // Block size = hidden_size (or multiple). We require hidden_size <= blockDim.x.
    // We'll use grid stride loop for hidden units.

    // For simplicity, each thread handles a subset of hidden units for all layers and time steps serialized over t.
    // But here one block per (b,t), each thread computes one hidden unit across all layers.
    // We need sequential dependencies across layers within a time step: layer0 output feeds layer1 input.
    // We can compute layer0 then layer1 using shared memory for intermediate h_t.
    // Shared memory needed: hidden_size * num_layers floats (h_t for each layer) and maybe c_t for each layer.
    // We'll allocate smem as: current_h for layer0, current_c for layer0, current_h for layer1, current_c for layer1.
    // But each thread only computes one hidden unit of each layer, so no thread cooperation needed for gates,
    // but we need h_t from previous time step and c_t from previous time step for recurrent connection.
    // We need to store h_{t-1} and c_{t-1} for all layers. We can store them in shared memory,
    // and each thread reads its corresponding hidden unit's previous state.
    // However, each block handles a single time step, and we need the h_{t-1} from previous time step,
    // which is not available in this block's shared memory. Alternative: use global memory with atomic? Too slow.

    // Better approach: Each block handles the entire sequence for one batch element, but that would be very heavy.
    // We'll use a simpler but correct approach: one block per batch, each thread computes one hidden unit for all time steps.
    // Then within a thread, sequential over time, and across layers we need care.

    // Let's redesign: Each block handles one batch and one hidden unit? Then we can have sequential time loop internally.
    // For a given hidden unit j and layer l, its h_t depends on previous h_{t-1} (same j) and c_{t-1} (same j),
    // but gate computations involve all hidden units from previous time step (for recurrent weights).
    // So we need the entire h_{t-1} vector, not just this unit. Therefore, one thread cannot compute one hidden unit independently
    // without access to full h_{t-1} from previous time step. We could store h_{t-1} in shared memory across threads.

    // So the approach: Use a block to process a batch element. Block threads = hidden_size (or multiple per hidden unit).
    // Shared memory stores h_t and c_t for current layer. Loop over time steps internally.
    // For each time step, compute gates for all hidden units using matrix-vector products.
    // For layer0, input is x[b,t,:] from global memory, Recurrent is h_{t-1} from shared.
    // For layer1, input is h_t^layer0 from shared, recurrent is h_{t-1}^layer1.
    // We need to compute 4 gates for each hidden unit: i, f, g, o.
    // Each thread can compute one hidden unit's gates by looping over input_size and hidden_size for layer0,
    // and over hidden_size and hidden_size for layer1. This is a lot of computation per thread but manageable.
    // We'll need to synchronize threads between time steps to get updated h_t.

    extern __shared__ float sdata[];
    float* h_prev_l0 = sdata;                        // hidden_size
    float* c_prev_l0 = sdata + hidden_size;          // hidden_size
    float* h_prev_l1 = sdata + 2 * hidden_size;      // hidden_size
    float* c_prev_l1 = sdata + 3 * hidden_size;      // hidden_size
    float* h_curr_l0 = sdata + 4 * hidden_size;      // hidden_size
    float* c_curr_l0 = sdata + 5 * hidden_size;      // hidden_size
    float* h_curr_l1 = sdata + 6 * hidden_size;      // hidden_size
    float* c_curr_l1 = sdata + 7 * hidden_size;      // hidden_size

    int b = blockIdx.x;  // batch index
    int tid = threadIdx.x;
    if (tid < hidden_size) {
        h_prev_l0[tid] = 0.0f;
        c_prev_l0[tid] = 0.0f;
        h_prev_l1[tid] = 0.0f;
        c_prev_l1[tid] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < seq_len; ++t) {
        // Compute layer0 gates for this time step.
        // Each thread computes hidden unit tid if tid < hidden_size, else idle.
        if (tid < hidden_size) {
            float i_gate, f_gate, g_gate, o_gate;
            // reset accumulators
            i_gate = (bias_ih_l0 ? bias_ih_l0[tid] : 0.0f) + (bias_hh_l0 ? bias_hh_l0[tid] : 0.0f);
            f_gate = (bias_ih_l0 ? bias_ih_l0[hidden_size + tid] : 0.0f) + (bias_hh_l0 ? bias_hh_l0[hidden_size + tid] : 0.0f);
            g_gate = (bias_ih_l0 ? bias_ih_l0[2*hidden_size + tid] : 0.0f) + (bias_hh_l0 ? bias_hh_l0[2*hidden_size + tid] : 0.0f);
            o_gate = (bias_ih_l0 ? bias_ih_l0[3*hidden_size + tid] : 0.0f) + (bias_hh_l0 ? bias_hh_l0[3*hidden_size + tid] : 0.0f);

            // Input-to-hidden contributions: x_t[b,t,:] dot weight_ih_l0[gate*bs+tid, :]
            const float* x_t = x + ((size_t)b * seq_len + t) * input_size;
            for (int k = 0; k < input_size; ++k) {
                float xk = x_t[k];
                i_gate += xk * weight_ih_l0[tid * input_size + k];
                f_gate += xk * weight_ih_l0[(hidden_size + tid) * input_size + k];
                g_gate += xk * weight_ih_l0[(2*hidden_size + tid) * input_size + k];
                o_gate += xk * weight_ih_l0[(3*hidden_size + tid) * input_size + k];
            }

            // Hidden-to-hidden contributions: h_prev_l0 dot weight_hh_l0[gate*bs+tid, :]
            for (int j = 0; j < hidden_size; ++j) {
                float hj = h_prev_l0[j];
                i_gate += hj * weight_hh_l0[tid * hidden_size + j];
                f_gate += hj * weight_hh_l0[(hidden_size + tid) * hidden_size + j];
                g_gate += hj * weight_hh_l0[(2*hidden_size + tid) * hidden_size + j];
                o_gate += hj * weight_hh_l0[(3*hidden_size + tid) * hidden_size + j];
            }

            float i_val = sigmoidf(i_gate);
            float f_val = sigmoidf(f_gate);
            float g_val = tanhf_approx(g_gate);
            float o_val = sigmoidf(o_gate);

            float c_val = f_val * c_prev_l0[tid] + i_val * g_val;
            float h_val = o_val * tanhf_approx(c_val);

            c_curr_l0[tid] = c_val;
            h_curr_l0[tid] = h_val;
        }
        __syncthreads();

        // Now layer1 uses h_curr_l0 as input.
        if (tid < hidden_size) {
            float i_gate, f_gate, g_gate, o_gate;
            i_gate = (bias_ih_l1 ? bias_ih_l1[tid] : 0.0f) + (bias_hh_l1 ? bias_hh_l1[tid] : 0.0f);
            f_gate = (bias_ih_l1 ? bias_ih_l1[hidden_size + tid] : 0.0f) + (bias_hh_l1 ? bias_hh_l1[hidden_size + tid] : 0.0f);
            g_gate = (bias_ih_l1 ? bias_ih_l1[2*hidden_size + tid] : 0.0f) + (bias_hh_l1 ? bias_hh_l1[2*hidden_size + tid] : 0.0f);
            o_gate = (bias_ih_l1 ? bias_ih_l1[3*hidden_size + tid] : 0.0f) + (bias_hh_l1 ? bias_hh_l1[3*hidden_size + tid] : 0.0f);

            // Input-to-hidden from layer0 output h_curr_l0 (size hidden_size)
            for (int k = 0; k < hidden_size; ++k) {
                float xk = h_curr_l0[k];
                i_gate += xk * weight_ih_l1[tid * hidden_size + k];
                f_gate += xk * weight_ih_l1[(hidden_size + tid) * hidden_size + k];
                g_gate += xk * weight_ih_l1[(2*hidden_size + tid) * hidden_size + k];
                o_gate += xk * weight_ih_l1[(3*hidden_size + tid) * hidden_size + k];
            }

            for (int j = 0; j < hidden_size; ++j) {
                float hj = h_prev_l1[j];
                i_gate += hj * weight_hh_l1[tid * hidden_size + j];
                f_gate += hj * weight_hh_l1[(hidden_size + tid) * hidden_size + j];
                g_gate += hj * weight_hh_l1[(2*hidden_size + tid) * hidden_size + j];
                o_gate += hj * weight_hh_l1[(3*hidden_size + tid) * hidden_size + j];
            }

            float i_val = sigmoidf(i_gate);
            float f_val = sigmoidf(f_gate);
            float g_val = tanhf_approx(g_gate);
            float o_val = sigmoidf(o_gate);

            float c_val = f_val * c_prev_l1[tid] + i_val * g_val;
            float h_val = o_val * tanhf_approx(c_val);

            c_curr_l1[tid] = c_val;
            h_curr_l1[tid] = h_val;
        }
        __syncthreads();

        // Write output for this time step: y[b,t,:] = h_curr_l1
        if (tid < hidden_size) {
            y[((size_t)b * seq_len + t) * hidden_size + tid] = h_curr_l1[tid];
        }

        // Update previous states for next time step.
        if (tid < hidden_size) {
            h_prev_l0[tid] = h_curr_l0[tid];
            c_prev_l0[tid] = c_curr_l0[tid];
            h_prev_l1[tid] = h_curr_l1[tid];
            c_prev_l1[tid] = c_curr_l1[tid];
        }
        __syncthreads();
    }

    // After all time steps, store final hidden and cell states.
    if (tid < hidden_size) {
        h_n[tid] = h_prev_l0[tid];               // layer0 h_n = position 0
        h_n[hidden_size + tid] = h_prev_l1[tid]; // layer1 h_n
        c_n[tid] = c_prev_l0[tid];
        c_n[hidden_size + tid] = c_prev_l1[tid];
    }
}

// Launcher for LSTM
extern "C" void lstm_launcher(
    const float* x,
    float* y,
    float* h_n,
    float* c_n,
    const float* weight_ih_l0,
    const float* weight_hh_l0,
    const float* bias_ih_l0,
    const float* bias_hh_l0,
    const float* weight_ih_l1,
    const float* weight_hh_l1,
    const float* bias_ih_l1,
    const float* bias_hh_l1,
    int batch_size,
    int seq_len,
    int input_size,
    int hidden_size,
    int num_layers,
    cudaStream_t stream
) {
    // Launch one block per batch element, with blockDim = hidden_size.
    // Shared memory required: 8 * hidden_size * sizeof(float) (for l0,l1 prev/curr states)
    int threads = hidden_size;  // must be <= 1024; hidden_size=256 in this task.
    size_t smem_size = 8 * hidden_size * sizeof(float);
    lstm_kernel<<<batch_size, threads, smem_size, stream>>>(
        x, y, h_n, c_n,
        weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0,
        weight_ih_l1, weight_hh_l1, bias_ih_l1, bias_hh_l1,
        batch_size, seq_len, input_size, hidden_size, num_layers);
}

// ------------------------- End of file -------------------------
