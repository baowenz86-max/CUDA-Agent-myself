#include <cuda_runtime.h>

// ============================================================================
// cat_along_dim CUDA kernel and launcher
// Copy tensor_a and tensor_b into contiguous output along dimension 1.
// Contract shapes: [num_layers, batch_size, hidden_size]
// Output shape:      [num_layers, 2*batch_size, hidden_size]
// All tensors are float32, contiguous. Output assumed preallocated contiguous.
// ============================================================================

__global__ void cat_along_dim_kernel(const float* __restrict__ tensor_a,
                                     const float* __restrict__ tensor_b,
                                     float* __restrict__ output,
                                     int64_t num_layers,
                                     int64_t batch_size,
                                     int64_t hidden_size)
{
    int64_t total_output = num_layers * 2 * batch_size * hidden_size;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_output) return;

    // Decompose output linear index
    int64_t hidden_idx = idx % hidden_size;
    int64_t b_h_index = idx / hidden_size;          // index within [num_layers * 2*batch_size]
    int64_t layer = b_h_index / (2 * batch_size);
    int64_t b2 = b_h_index % (2 * batch_size);      // position along concatenated batch dim

    int64_t src_lane;
    const float* src;
    if (b2 < batch_size) {
        // from tensor_a: layer, batch=b2, hidden_idx
        src_lane = (layer * batch_size + b2) * hidden_size + hidden_idx;
        src = tensor_a;
    } else {
        // from tensor_b: layer, batch=b2-batch_size, hidden_idx
        src_lane = (layer * batch_size + (b2 - batch_size)) * hidden_size + hidden_idx;
        src = tensor_b;
    }
    output[idx] = src[src_lane];
}

// Launcher for cat_along_dim, for binding layer.
// Input tensors must be on current CUDA stream, contiguous, float32.
extern "C" void cat_along_dim_forward_launcher(const float* tensor_a,
                                               const float* tensor_b,
                                               float* output,
                                               int64_t num_layers,
                                               int64_t batch_size,
                                               int64_t hidden_size,
                                               cudaStream_t stream)
{
    int64_t total_output = num_layers * 2 * batch_size * hidden_size;
    const int block_size = 256;
    int64_t grid_size = (total_output + block_size - 1) / block_size;
    cat_along_dim_kernel<<<grid_size, block_size, 0, stream>>>(
        tensor_a, tensor_b, output, num_layers, batch_size, hidden_size);
}

// ============================================================================
// lstm_forward launcher placeholder
// NOTE: The contract requires lstm_forward to use cuDNN. This .cu file cannot
// include cuDNN or C++ headers, so the actual implementation will be added in
// the C++ binding phase. This declaration keeps the symbol stable; it will
// currently do nothing and returns success, which is insufficient. Binding
// layer must NOT call this placeholder before cuDNN implementation exists.
// ============================================================================

extern "C" void lstm_forward_launcher(const float* input,
                                      const float* weight_ih_l0,
                                      const float* weight_hh_l0,
                                      const float* bias_ih_l0,
                                      const float* bias_hh_l0,
                                      const float* weight_ih_l1,
                                      const float* weight_hh_l1,
                                      const float* bias_ih_l1,
                                      const float* bias_hh_l1,
                                      float* output,
                                      float* hidden,
                                      float* cell,
                                      int64_t batch_size,
                                      int64_t seq_len,
                                      int64_t input_size,
                                      int64_t hidden_size,
                                      int64_t num_layers,
                                      bool batch_first,
                                      cudaStream_t stream)
{
    // Placeholder: no computation. Do not call this before C++ cuDNN phase.
}
