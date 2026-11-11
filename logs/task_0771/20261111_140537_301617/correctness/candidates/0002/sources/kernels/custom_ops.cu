#include <cuda_runtime.h>
#include <stdint.h>

// -----------------------------------------------------------------------------
// LSTM forward kernel for a single layer.
// Computes output, hidden state, and cell state for all time steps.
// Each thread handles one hidden unit for one batch sample across all time steps.
// blockDim.x = hidden_size, gridDim.x = batch.
// Shared memory holds h_prev and c_prev for the batch sample.
// Synchronization is critical to avoid race conditions.
// -----------------------------------------------------------------------------

template <int HIDDEN_SIZE>
__global__ void lstm_layer_kernel_smem(
    const float* __restrict__ x,       // [batch, seq_len, input_size_layer]
    float* __restrict__ y,             // [batch, seq_len, HIDDEN_SIZE]
    float* __restrict__ hn,            // [batch, HIDDEN_SIZE] final hidden state (for this layer)
    float* __restrict__ cn,            // [batch, HIDDEN_SIZE] final cell state
    const float* __restrict__ W_ih,    // [4*HIDDEN_SIZE, input_size_layer]
    const float* __restrict__ W_hh,    // [4*HIDDEN_SIZE, HIDDEN_SIZE]
    const float* __restrict__ b_ih,    // [4*HIDDEN_SIZE]
    const float* __restrict__ b_hh,    // [4*HIDDEN_SIZE]
    int batch,
    int seq_len,
    int input_size_layer
) {
    __shared__ float h_prev[HIDDEN_SIZE];
    __shared__ float c_prev[HIDDEN_SIZE];

    int batch_idx = blockIdx.x;
    int unit = threadIdx.x;

    // Initialize shared h and c to zero
    if (unit < HIDDEN_SIZE) {
        h_prev[unit] = 0.0f;
        c_prev[unit] = 0.0f;
    }
    __syncthreads();

    const float* x_batch = x + (size_t)batch_idx * seq_len * input_size_layer;
    float* y_batch = y + (size_t)batch_idx * seq_len * HIDDEN_SIZE;

    for (int t = 0; t < seq_len; ++t) {
        const float* x_t = x_batch + (size_t)t * input_size_layer;

        // Compute pre-activations
        float pre_i = b_ih[0 * HIDDEN_SIZE + unit] + b_hh[0 * HIDDEN_SIZE + unit];
        float pre_f = b_ih[1 * HIDDEN_SIZE + unit] + b_hh[1 * HIDDEN_SIZE + unit];
        float pre_g = b_ih[2 * HIDDEN_SIZE + unit] + b_hh[2 * HIDDEN_SIZE + unit];
        float pre_o = b_ih[3 * HIDDEN_SIZE + unit] + b_hh[3 * HIDDEN_SIZE + unit];

        // Input contribution
        for (int j = 0; j < input_size_layer; ++j) {
            float xj = x_t[j];
            pre_i += W_ih[(0 * HIDDEN_SIZE + unit) * input_size_layer + j] * xj;
            pre_f += W_ih[(1 * HIDDEN_SIZE + unit) * input_size_layer + j] * xj;
            pre_g += W_ih[(2 * HIDDEN_SIZE + unit) * input_size_layer + j] * xj;
            pre_o += W_ih[(3 * HIDDEN_SIZE + unit) * input_size_layer + j] * xj;
        }

        // Recurrent contribution using shared h_prev
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            float hj = h_prev[j];
            pre_i += W_hh[(0 * HIDDEN_SIZE + unit) * HIDDEN_SIZE + j] * hj;
            pre_f += W_hh[(1 * HIDDEN_SIZE + unit) * HIDDEN_SIZE + j] * hj;
            pre_g += W_hh[(2 * HIDDEN_SIZE + unit) * HIDDEN_SIZE + j] * hj;
            pre_o += W_hh[(3 * HIDDEN_SIZE + unit) * HIDDEN_SIZE + j] * hj;
        }

        // Activations
        float i_gate = 1.0f / (1.0f + expf(-pre_i));
        float f_gate = 1.0f / (1.0f + expf(-pre_f));
        float g_gate = tanhf(pre_g);
        float o_gate = 1.0f / (1.0f + expf(-pre_o));

        float c_new = f_gate * c_prev[unit] + i_gate * g_gate;
        float h_new = o_gate * tanhf(c_new);

        // Store output for this time step
        y_batch[(size_t)t * HIDDEN_SIZE + unit] = h_new;

        // Synchronize before updating shared state (ensure all threads have read old h_prev and c_prev)
        __syncthreads();

        // Update shared state for next time step
        h_prev[unit] = h_new;
        c_prev[unit] = c_new;

        // Synchronize to make new state visible to all threads before next iteration
        __syncthreads();
    }

    // After processing all time steps, write final h and c to global memory
    hn[batch_idx * HIDDEN_SIZE + unit] = h_prev[unit];
    cn[batch_idx * HIDDEN_SIZE + unit] = c_prev[unit];
}

void launch_lstm_layer(
    const float* x,
    float* y,
    float* hn,
    float* cn,
    const float* W_ih,
    const float* W_hh,
    const float* b_ih,
    const float* b_hh,
    int batch,
    int seq_len,
    int input_size_layer,
    int hidden_size,
    cudaStream_t stream
) {
    dim3 block(hidden_size);
    dim3 grid(batch);

    switch (hidden_size) {
        case 16:
            lstm_layer_kernel_smem<16><<<grid, block, 0, stream>>>(
                x, y, hn, cn, W_ih, W_hh, b_ih, b_hh, batch, seq_len, input_size_layer);
            break;
        case 32:
            lstm_layer_kernel_smem<32><<<grid, block, 0, stream>>>(
                x, y, hn, cn, W_ih, W_hh, b_ih, b_hh, batch, seq_len, input_size_layer);
            break;
        case 64:
            lstm_layer_kernel_smem<64><<<grid, block, 0, stream>>>(
                x, y, hn, cn, W_ih, W_hh, b_ih, b_hh, batch, seq_len, input_size_layer);
            break;
        case 128:
            lstm_layer_kernel_smem<128><<<grid, block, 0, stream>>>(
                x, y, hn, cn, W_ih, W_hh, b_ih, b_hh, batch, seq_len, input_size_layer);
            break;
        default:
            // Unsupported hidden size; no launch.
            break;
    }
}

extern "C" void launch_custom_lstm_forward(
    const float* input,
    float* output,
    float* hn,
    float* cn,
    float* temp_output,
    const float* W_ih_l0,
    const float* W_hh_l0,
    const float* b_ih_l0,
    const float* b_hh_l0,
    const float* W_ih_l1,
    const float* W_hh_l1,
    const float* b_ih_l1,
    const float* b_hh_l1,
    int batch,
    int seq_len,
    int input_size,
    int hidden_size,
    cudaStream_t stream
) {
    // First layer: input_size_layer = input_size
    launch_lstm_layer(input, temp_output, hn, cn, W_ih_l0, W_hh_l0, b_ih_l0, b_hh_l0,
                      batch, seq_len, input_size, hidden_size, stream);

    // Second layer: input_size_layer = hidden_size
    // Note: hn and cn are contiguous [2, batch, hidden_size]. For layer 1, offset by batch*hidden_size.
    launch_lstm_layer(temp_output, output, hn + batch * hidden_size, cn + batch * hidden_size,
                      W_ih_l1, W_hh_l1, b_ih_l1, b_hh_l1,
                      batch, seq_len, hidden_size, hidden_size, stream);
}

__global__ void isin_kernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ values,
    bool* __restrict__ mask,
    int num_values,
    int64_t total_elements
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float val = input[idx];
    bool found = false;
    for (int v = 0; v < num_values; ++v) {
        if (val == (float)values[v]) {
            found = true;
            break;
        }
    }
    mask[idx] = found;
}

void launch_isin_forward(
    const float* input,
    const int64_t* values,
    bool* mask,
    int num_values,
    int64_t total_elements,
    cudaStream_t stream
) {
    const int threads = 256;
    int64_t blocks = (total_elements + threads - 1) / threads;
    isin_kernel<<<blocks, threads, 0, stream>>>(input, values, mask, num_values, total_elements);
}

extern "C" void custom_isin_launcher(
    const float* input,
    const int64_t* values,
    bool* mask,
    int num_values,
    int64_t total_elements,
    cudaStream_t stream
) {
    launch_isin_forward(input, values, mask, num_values, total_elements, stream);
}
