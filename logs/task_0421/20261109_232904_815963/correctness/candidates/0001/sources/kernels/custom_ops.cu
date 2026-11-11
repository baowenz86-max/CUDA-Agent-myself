// CUDA kernels and launchers for the custom ops required by model_new.py
// embedding_forward: look up embeddings from an int64 index tensor
// bitwise_xor_int64: float32 -> int64 truncation, bitwise XOR with broadcasting, -> float32

#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// embedding_forward
// -----------------------------------------------------------------------------
// Each input index maps to a row of the weight tensor.
// input shape: [batch_size, sequence_length], int64
// weight shape: [num_embeddings, embedding_dim], float32
// output shape: [batch_size, sequence_length, embedding_dim], float32, contiguous
__global__ void embedding_lookup_kernel(
    const int64_t* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int total_indices,
    int embedding_dim,
    int64_t num_embeddings) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_indices) return;

    int64_t token = input[idx];
    // PyTorch Embedding without padding: out-of-range indices are undefined, but
    // avoid illegal memory access by clamping (behavior matches typical undefined read).
    if (token < 0 || token >= num_embeddings) {
        token = 0;  // safe fallback, results are not guaranteed for invalid indices
    }

    const float* src = weight + token * embedding_dim;
    float* dst = output + (int64_t)idx * embedding_dim;
    for (int d = 0; d < embedding_dim; ++d) {
        dst[d] = src[d];
    }
}

extern "C" void embedding_forward_launcher(
    const int64_t* input,
    const float* weight,
    float* output,
    int batch_size,
    int sequence_length,
    int embedding_dim,
    int64_t num_embeddings,
    cudaStream_t stream) {
    int total_indices = batch_size * sequence_length;
    if (total_indices <= 0 || embedding_dim <= 0) {
        return;
    }

    const int block_size = 256;
    int grid_size = (total_indices + block_size - 1) / block_size;
    embedding_lookup_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weight, output, total_indices, embedding_dim, num_embeddings);
}

// -----------------------------------------------------------------------------
// bitwise_xor_int64
// -----------------------------------------------------------------------------
// Both x and y are float32 in memory, but the reference first converts to int64
// with truncation, then performs bitwise XOR on the int64 bits, then converts
// back to float32.
// x shape: [batch_size, sequence_length, embedding_dim]
// y shape: [sequence_length, embedding_dim]
// Output shape: [batch_size, sequence_length, embedding_dim]
__global__ void bitwise_xor_int64_kernel(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ output,
    int total_elements,
    int embedding_dim,
    int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Convert x element to int64 via truncation (C-style cast from float to int64
    // truncates toward zero, matching PyTorch .to(torch.int64)).
    int64_t xi = static_cast<int64_t>(x[idx]);

    // Compute broadcast index into y: y has shape [seq_len, embedding_dim].
    // The linear idx in x is (b * seq_len + s) * emb_dim + d.
    int s = (idx / embedding_dim) % seq_len;
    int d = idx % embedding_dim;
    int64_t yi = static_cast<int64_t>(y[s * embedding_dim + d]);

    int64_t res = xi ^ yi;
    output[idx] = static_cast<float>(res);
}

extern "C" void bitwise_xor_int64_launcher(
    const float* x,
    const float* y,
    float* output,
    int batch_size,
    int sequence_length,
    int embedding_dim,
    cudaStream_t stream) {
    int total_elements = batch_size * sequence_length * embedding_dim;
    if (total_elements <= 0) {
        return;
    }

    const int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    bitwise_xor_int64_kernel<<<grid_size, block_size, 0, stream>>>(
        x, y, output, total_elements, embedding_dim, sequence_length);
}
