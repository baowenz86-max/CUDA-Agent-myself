// Elementwise addition of positional encoding: out = x + pos_enc[:, :seq_len, :]
// x: [batch, seq_len, embed_dim], pos_enc: [1, pos_dim, embed_dim], output same as x.
// Only float32, contiguous, CUDA.

#include <cuda_runtime.h>

__global__ void add_pos_enc_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ pos_enc,
    int batch_size,
    int seq_len,
    int embed_dim,
    int pos_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * seq_len * embed_dim;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int embed = i % embed_dim;
        int seq = (i / embed_dim) % seq_len;
        int batch = i / (embed_dim * seq_len);
        // pos_enc is [1, pos_dim, embed_dim], we use seq index up to seq_len, which must be <= pos_dim (true in this model)
        float pe = pos_enc[seq * embed_dim + embed]; // since batch dim is 1, offset seq*embed_dim+embed
        output[i] = input[i] + pe;
    }
}

extern "C" void add_pos_enc_launcher(
    float* output,
    const float* input,
    const float* pos_enc,
    int batch_size,
    int seq_len,
    int embed_dim,
    int pos_dim,
    cudaStream_t stream
) {
    int total = batch_size * seq_len * embed_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_pos_enc_kernel<<<blocks, threads, 0, stream>>>(
        output, input, pos_enc, batch_size, seq_len, embed_dim, pos_dim);
}
