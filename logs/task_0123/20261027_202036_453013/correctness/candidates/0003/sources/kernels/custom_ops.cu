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
// Each block handles one output element? Or one row. We'll do one thread per output element.
__global__ void linear_kernel(const float* input, const float* weight, const float* bias,
                              float* output, int batch, int in_features, int out_features) {
    int row = blockIdx.x;       // batch index
    int col = blockIdx.y;       // out_features index
    if (row >= batch || col >= out_features) return;
    int tid = threadIdx.x;      // reduce over in_features
    __shared__ float sum[256];  // blockDim.x <= 256 assumed
    float val = 0.0f;
    for (int k = tid; k < in_features; k += blockDim.x) {
        val += input[row * in_features + k] * weight[col * in_features + k];
    }
    sum[tid] = val;
    __syncthreads();
    // reduction within block
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

// LSTM kernel: we implement a naive sequential LSTM for 2 layers.
// Input: (seq_len, batch, input_size), weights per layer: W_ih (4*hidden, input), W_hh (4*hidden, hidden), b_ih, b_hh.
// Output: (seq_len, batch, hidden) for last layer.
// Hidden state per layer: h_t, c_t of size (batch, hidden)
// We'll launch one block per (layer, batch?) but sequential over time is needed.
// Simpler: one block per layer? No, we need to process entire sequence; easiest is to have one block per layer and batch? Actually each time step depends on previous.
// Approach: Use a single kernel that loops over seq_len and layers, with one block per batch element? That would require synchronization across batch? Not needed because hidden states are per batch element independently.
// We can have one block per batch element, and inside that block process all layers and all time steps sequentially. That requires each block to have all weights for all layers in registers/shared? We could load weights on the fly but that's slow. Given small sizes (hidden=30, batch=16), it's feasible: each block (one per batch element) can have all weights in shared memory (or registers if they fit). We'll use a block of 256 threads, but reduce to 30 threads for compute? Better: one block per batch element, with hidden_size*4 maybe? But we need to compute matrix multiplications of sizes up to 4*hidden x input, small enough to do with one warp? We'll do a simple kernel: each block processes one batch element, loops over time, and within each time loops over layers. For each layer, it computes gates using threads (each thread computes one output element of the gates? that's 4*hidden elements, could be done by blockDim.x threads). Since hidden=30, 120 gates, block of 128 or 256 threads can compute them, but need reductions over input/hidden dimensions. That's inefficient but manageable.
// For simplicity and correctness, we can implement a straightforward kernel where each block handles one batch element, uses all threads to compute each gate's dot products, but with small sizes it may be okay.
// Alternative: use cuBLAS if available, but we assume not. We will implement our own GEMM for LSTM within the kernel.
// This code is for fix; we'll do a basic implementation with shared memory for weights and states.

// Constants: hidden_size <= 64? Actual hidden=30, input=20, batch=16. We'll hardcode max sizes via template? We'll use dynamic but assume max hidden <= 128, input <= 128, batch <= 64 to fit shared memory.

__global__ void lstm_kernel(const float* input,  // (seq_len, batch, input_size)
                            const float* W_ih_l0, const float* W_hh_l0,
                            const float* b_ih_l0, const float* b_hh_l0,
                            const float* W_ih_l1, const float* W_hh_l1,
                            const float* b_ih_l1, const float* b_hh_l1,
                            float* output,  // (seq_len, batch, hidden)
                            int seq_len, int batch, int input_size, int hidden_size) {
    // One block per batch element
    int b = blockIdx.x;
    if (b >= batch) return;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Shared memory for weights and biases for both layers, and states.
    // We'll assume hidden_size <= 64, input_size <= 64, gate_size = 4*hidden <= 256, batch <= 64.
    extern __shared__ float smem[];
    // Layout: 
    // W_ih_l0 [4*hidden * input]
    // W_hh_l0 [4*hidden * hidden]
    // b_ih_l0 [4*hidden]
    // b_hh_l0 [4*hidden]
    // W_ih_l1 [4*hidden * hidden]
    // W_hh_l1 [4*hidden * hidden]
    // b_ih_l1 [4*hidden]
    // b_hh_l1 [4*hidden]
    // h0 [hidden]
    // c0 [hidden]
    // h1 [hidden]
    // c1 [hidden]
    // temporary gate vectors [4*hidden]
    int offset = 0;
    float* s_W_ih_l0 = smem + offset; offset += 4 * hidden_size * input_size;
    float* s_W_hh_l0 = smem + offset; offset += 4 * hidden_size * hidden_size;
    float* s_b_ih_l0 = smem + offset; offset += 4 * hidden_size;
    float* s_b_hh_l0 = smem + offset; offset += 4 * hidden_size;
    float* s_W_ih_l1 = smem + offset; offset += 4 * hidden_size * hidden_size;
    float* s_W_hh_l1 = smem + offset; offset += 4 * hidden_size * hidden_size;
    float* s_b_ih_l1 = smem + offset; offset += 4 * hidden_size;
    float* s_b_hh_l1 = smem + offset; offset += 4 * hidden_size;
    float* s_h0 = smem + offset; offset += hidden_size;
    float* s_c0 = smem + offset; offset += hidden_size;
    float* s_h1 = smem + offset; offset += hidden_size;
    float* s_c1 = smem + offset; offset += hidden_size;
    float* s_gates = smem + offset; // 4*hidden_size

    // Load weights and biases into shared memory (cooperatively)
    int total_weights = 4 * hidden_size * (input_size + hidden_size + 1) * 2;  // both layers
    for (int i = tid; i < total_weights; i += block_size) {
        // decode index
        if (i < 4 * hidden_size * input_size) {
            s_W_ih_l0[i] = W_ih_l0[i];
        } else if (i < 4 * hidden_size * (input_size + hidden_size)) {
            int j = i - 4 * hidden_size * input_size;
            s_W_hh_l0[j] = W_hh_l0[j];
        } else if (i < 4 * hidden_size * (input_size + hidden_size) + 4 * hidden_size) {
            int j = i - 4 * hidden_size * (input_size + hidden_size);
            s_b_ih_l0[j] = b_ih_l0[j];
            s_b_hh_l0[j] = b_hh_l0[j];
        } else if (i < 4 * hidden_size * (input_size + hidden_size) + 4 * hidden_size * 2) {
            int j = i - (4 * hidden_size * (input_size + hidden_size) + 4 * hidden_size);
            s_b_hh_l0[j] = b_hh_l0[j];
        } else if (i < 4 * hidden_size * (input_size + hidden_size) + 4 * hidden_size * 2 + 4 * hidden_size * hidden_size) {
            int j = i - (4 * hidden_size * (input_size + hidden_size) + 4 * hidden_size * 2);
            s_W_ih_l1[j] = W_ih_l1[j];
        } else if (i < 4 * hidden_size * (input_size + hidden_size) + 4 * hidden_size * 2 + 4 * hidden_size * hidden_size * 2) {
            int j = i - (4 * hidden_size * (input_size + hidden_size) + 4 * hidden_size * 2 + 4 * hidden_size * hidden_size);
            s_W_hh_l1[j] = W_hh_l1[j];
        } else if (i < 4 * hidden_size * (input_size + hidden_size) + 4 * hidden_size * 2 + 4 * hidden_size * hidden_size * 2 + 4 * hidden_size) {
            int j = i - (4 * hidden_size * (input_size + hidden_size) + 4 * hidden_size * 2 + 4 * hidden_size * hidden_size * 2);
            s_b_ih_l1[j] = b_ih_l1[j];
        } else {
            int j = i - (4 * hidden_size * (input_size + hidden_size) + 4 * hidden_size * 2 + 4 * hidden_size * hidden_size * 2 + 4 * hidden_size);
            s_b_hh_l1[j] = b_hh_l1[j];
        }
    }
    __syncthreads();

    // Initialize hidden states to zero
    for (int i = tid; i < hidden_size; i += block_size) {
        s_h0[i] = 0.0f;
        s_c0[i] = 0.0f;
        s_h1[i] = 0.0f;
        s_c1[i] = 0.0f;
    }
    __syncthreads();

    // Process each time step sequentially
    for (int t = 0; t < seq_len; t++) {
        // Layer 0
        // Compute gates: i, f, g, o = W_ih * x_t + b_ih + W_hh * h_prev + b_hh
        // x_t is input[t * batch * input_size + b * input_size + :]
        const float* x_t = input + t * batch * input_size + b * input_size;
        // We need to compute 4*hidden outputs of gates. Each thread computes one gate element? That would require block_size >= 4*hidden (120). We'll use block_size threads, each computes a subset of gate elements, with reduction over input_size+hidden_size.
        // For simplicity, each thread computes one gate element at a time, but that's slow but hidden=30, 4*30=120, block_size=256, so plenty.
        // We'll have each thread compute one gate element if tid < 4*hidden_size.
        if (tid < 4 * hidden_size) {
            int gate_idx = tid;  // 0..119
            float val = 0.0f;
            // Add W_ih * x_t
            for (int k = 0; k < input_size; k++) {
                val += s_W_ih_l0[gate_idx * input_size + k] * x_t[k];
            }
            // Add b_ih
            val += s_b_ih_l0[gate_idx];
            // Add W_hh * h_prev
            for (int k = 0; k < hidden_size; k++) {
                val += s_W_hh_l0[gate_idx * hidden_size + k] * s_h0[k];
            }
            // Add b_hh
            val += s_b_hh_l0[gate_idx];
            s_gates[gate_idx] = val;
        }
        __syncthreads();

        // Apply activations and update c, h
        if (tid < hidden_size) {
            int i = tid;
            float ig = 1.0f / (1.0f + expf(-s_gates[i]));  // sigmoid
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
        // Compute gates for layer 1
        if (tid < 4 * hidden_size) {
            int gate_idx = tid;
            float val = 0.0f;
            // W_ih_l1 * h0
            for (int k = 0; k < hidden_size; k++) {
                val += s_W_ih_l1[gate_idx * hidden_size + k] * s_h0[k];
            }
            val += s_b_ih_l1[gate_idx];
            // W_hh_l1 * h1_prev
            for (int k = 0; k < hidden_size; k++) {
                val += s_W_hh_l1[gate_idx * hidden_size + k] * s_h1[k];
            }
            val += s_b_hh_l1[gate_idx];
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

        // Write output for this time step: output[t * batch * hidden_size + b * hidden_size + i] = h1[i]
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
    // Calculate shared memory size
    int smem_size = 0;
    smem_size += 4 * hidden_size * input_size * sizeof(float);
    smem_size += 4 * hidden_size * hidden_size * sizeof(float);
    smem_size += 4 * hidden_size * sizeof(float);  // b_ih_l0
    smem_size += 4 * hidden_size * sizeof(float);  // b_hh_l0
    smem_size += 4 * hidden_size * hidden_size * sizeof(float);  // W_ih_l1
    smem_size += 4 * hidden_size * hidden_size * sizeof(float);  // W_hh_l1
    smem_size += 4 * hidden_size * sizeof(float);  // b_ih_l1
    smem_size += 4 * hidden_size * sizeof(float);  // b_hh_l1
    smem_size += hidden_size * sizeof(float) * 4; // h0,c0,h1,c1
    smem_size += 4 * hidden_size * sizeof(float); // gates
    lstm_kernel<<<blocks, threads, smem_size, stream>>>(
        input, W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0,
        W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1,
        output, seq_len, batch, input_size, hidden_size);
}
