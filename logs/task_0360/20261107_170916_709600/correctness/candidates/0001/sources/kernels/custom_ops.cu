#include <cuda_runtime.h>

// CUDA kernel for one RNN layer with tanh activation.
// Each thread computes one hidden unit for a given batch element and time step.
// The kernel iterates over all time steps sequentially, writing output for each time step.
// Inputs:
//   input: input activations for this layer, shape [seq_length, batch, input_size]
//   output: output activations, shape [seq_length, batch, hidden_size]
//   weight_ih: shape [hidden_size, input_size]
//   weight_hh: shape [hidden_size, hidden_size]
//   bias_ih: optional, shape [hidden_size]
//   bias_hh: optional, shape [hidden_size]
//   h_last: final hidden state for this layer, shape [batch, hidden_size]
//   seq_length, batch, input_size, hidden_size
__global__ void rnn_cell_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight_ih,
    const float* __restrict__ weight_hh,
    const float* __restrict__ bias_ih,
    const float* __restrict__ bias_hh,
    float* __restrict__ h_last,
    int seq_length,
    int batch,
    int input_size,
    int hidden_size
) {
    int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hidden_idx >= hidden_size) return;
    int batch_idx = blockIdx.y; // batch dimension in grid y

    // initial hidden state is zero
    float h = 0.0f;

    // base precomputation: bias_ih[hidden_idx] + bias_hh[hidden_idx]
    float total_bias = 0.0f;
    if (bias_ih != nullptr) total_bias += bias_ih[hidden_idx];
    if (bias_hh != nullptr) total_bias += bias_hh[hidden_idx];

    for (int t = 0; t < seq_length; ++t) {
        // gate = bias + sum_i input[t,batch,i] * weight_ih[hidden_idx,i] + sum_j h_prev[j] * weight_hh[hidden_idx,j]
        float gate = total_bias;

        const float* x_t = input + (size_t)t * batch * input_size + (size_t)batch_idx * input_size;
        for (int i = 0; i < input_size; ++i) {
            gate += x_t[i] * weight_ih[hidden_idx * input_size + i];
        }

        // For recurrence, we need the previous hidden state vector h_prev.
        // Since each thread only maintains its own h scalar, we need access to all h_j from previous time step.
        // We cannot simply use a local scalar. We need to store h_prev for all hidden units in a shared or global buffer.
        // Solution: store hidden state for all units in a global buffer per batch and time step.
        // For simplicity, we can maintain a global array h_state of shape [batch, hidden] updated each time step.
        // However, inside this kernel we need communication between threads of different hidden_idx.
        // Use shared memory to store previous hidden state for the whole block.
        // Each block handles one batch element and all hidden units. hidden_size is 64, so one block with 64 threads is enough.
    }
}
