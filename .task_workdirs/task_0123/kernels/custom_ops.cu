#include <cuda_runtime.h>
#include <cstdint>

// Exp kernel (existing)
__global__ void exp_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        output[i] = expf(input[i]);
    }
}

extern "C" void exp_forward_launcher(const float* input, float* output, int n, cudaStream_t stream) {
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks == 0) return;
    exp_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

// Linear kernel: y = x * W^T + b
// x: (batch, in_features), W: (out_features, in_features), b: (out_features,)
__global__ void linear_kernel(const float* input, const float* weight, const float* bias,
                              float* output, int batch, int in_features, int out_features) {
    int row = blockIdx.x;
    int col = blockIdx.y;
    if (row >= batch || col >= out_features) return;
    int tid = threadIdx.x;
    __shared__ float sum[256];
    float val = 0.0f;
    for (int k = tid; k < in_features; k += blockDim.x) {
        val += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum[tid] = val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum[tid] += sum[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[row * out_features + col] = sum[0] + bias[col];
    }
}

extern "C" void linear_launcher(const float* input, const float* weight, const float* bias,
                                float* output, int batch, int in_features, int out_features,
                                cudaStream_t stream) {
    dim3 grid(batch, out_features);
    const int threads = 256;
    linear_kernel<<<grid, threads, 0, stream>>>(input, weight, bias, output, batch, in_features, out_features);
}

// LSTM kernel: one block per batch element, no large shared memory for weights.
// Weights are read directly from global memory. Shared memory is used only for
// temporary gate values and hidden/cell states for the current time step.
__global__ void lstm_kernel(const float* input,  // (seq_len, batch, input_size)
                            const float* W_ih_l0, const float* W_hh_l0,
                            const float* b_ih_l0, const float* b_hh_l0,
                            const float* W_ih_l1, const float* W_hh_l1,
                            const float* b_ih_l1, const float* b_hh_l1,
                            float* output,         // (seq_len, batch, hidden_size)
                            int seq_len, int batch, int input_size, int hidden_size) {
    int b = blockIdx.x;
    if (b >= batch) return;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    extern __shared__ float smem[];
    // Layout: gates (4*hidden), then states h0,c0,h1,c1 (4*hidden)
    float* s_gates = smem;
    float* s_h0 = smem + 4 * hidden_size;
    float* s_c0 = smem + 5 * hidden_size;
    float* s_h1 = smem + 6 * hidden_size;
    float* s_c1 = smem + 7 * hidden_size;

    // Initialize states to zero
    if (tid < hidden_size) {
        s_h0[tid] = 0.0f;
        s_c0[tid] = 0.0f;
        s_h1[tid] = 0.0f;
        s_c1[tid] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < seq_len; t++) {
        // Layer 0
        // Compute gates for layer 0: 4*hidden values
        if (tid < 4 * hidden_size) {
            int gate_idx = tid;
            float val = 0.0f;
            const float* x_t = input + t * batch * input_size + b * input_size;
            // W_ih * x_t
            for (int k = 0; k < input_size; k++) {
                val += W_ih_l0[gate_idx * input_size + k] * x_t[k];
            }
            val += b_ih_l0[gate_idx];
            // W_hh * h0_prev
            for (int k = 0; k < hidden_size; k++) {
                val += W_hh_l0[gate_idx * hidden_size + k] * s_h0[k];
            }
            val += b_hh_l0[gate_idx];
            s_gates[gate_idx] = val;
        }
        __syncthreads();

        // Update c0, h0
        if (tid < hidden_size) {
            int i = tid;
            float ig = 1.0f / (1.0f + expf(-s_gates[i]));
            float fg = 1.0f / (1.0f + expf(-s_gates[hidden_size + i]));
            float gg = tanhf(s_gates[2 * hidden_size + i]);
            float og = 1.0f / (1.0f + expf(-s_gates[3 * hidden_size + i]));
            float c_new = fg * s_c0[i] + ig * gg;
            float h_new = og * tanhf(c_new);
            s_c0[i] = c_new;
            s_h0[i] = h_new;
        }
        __syncthreads();

        // Layer 1 uses h0 as input
        if (tid < 4 * hidden_size) {
            int gate_idx = tid;
            float val = 0.0f;
            // W_ih_l1 * h0
            for (int k = 0; k < hidden_size; k++) {
                val += W_ih_l1[gate_idx * hidden_size + k] * s_h0[k];
            }
            val += b_ih_l1[gate_idx];
            // W_hh_l1 * h1_prev
            for (int k = 0; k < hidden_size; k++) {
                val += W_hh_l1[gate_idx * hidden_size + k] * s_h1[k];
            }
            val += b_hh_l1[gate_idx];
            s_gates[gate_idx] = val;
        }
        __syncthreads();

        // Update c1, h1
        if (tid < hidden_size) {
            int i = tid;
            float ig = 1.0f / (1.0f + expf(-s_gates[i]));
            float fg = 1.0f / (1.0f + expf(-s_gates[hidden_size + i]));
            float gg = tanhf(s_gates[2 * hidden_size + i]);
            float og = 1.0f / (1.0f + expf(-s_gates[3 * hidden_size + i]));
            float c_new = fg * s_c1[i] + ig * gg;
            float h_new = og * tanhf(c_new);
            s_c1[i] = c_new;
            s_h1[i] = h_new;
        }
        __syncthreads();

        // Write output for this time step
        if (tid < hidden_size) {
            output[t * batch * hidden_size + b * hidden_size + tid] = s_h1[tid];
        }
        __syncthreads();
    }
}

extern "C" void lstm_launcher(const float* input,
                               const float* W_ih_l0, const float* W_hh_l0,
                               const float* b_ih_l0, const float* b_hh_l0,
                               const float* W_ih_l1, const float* W_hh_l1,
                               const float* b_ih_l1, const float* b_hh_l1,
                               float* output,
                               int seq_len, int batch, int input_size, int hidden_size,
                               cudaStream_t stream) {
    int threads = 256;
    int blocks = batch;
    // Shared memory: gates (4*hidden) + states (4*hidden) = 8*hidden floats
    int smem_size = 8 * hidden_size * sizeof(float);
    lstm_kernel<<<blocks, threads, smem_size, stream>>>(
        input, W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0,
        W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1,
        output, seq_len, batch, input_size, hidden_size);
}
