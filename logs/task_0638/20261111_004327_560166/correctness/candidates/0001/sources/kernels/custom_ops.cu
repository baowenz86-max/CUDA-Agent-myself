#include <cuda_runtime.h>
#include <cstdint>

extern "C" {

__global__ void embedding_gather_kernel(
    float* __restrict__ output,
    const int64_t* __restrict__ input,
    const float* __restrict__ weight,
    int batch,
    int seq_len,
    int embedding_dim,
    int num_embeddings
) {
    int total = batch * seq_len * embedding_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int k = i % embedding_dim;
        int pos = i / embedding_dim;
        int j = pos % seq_len;
        int b = pos / seq_len;
        int64_t index = input[b * seq_len + j];
        if (index >= 0 && index < num_embeddings) {
            output[i] = weight[index * embedding_dim + k];
        } else {
            output[i] = 0.0f;
        }
    }
}

__global__ void add_tensors_kernel(
    float* __restrict__ output,
    const float* __restrict__ a,
    const float* __restrict__ b,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        output[i] = a[i] + b[i];
    }
}

void embedding1_forward_launcher(
    float* output,
    const int64_t* input,
    const float* weight,
    int batch,
    int seq_len,
    int embedding_dim,
    int num_embeddings,
    cudaStream_t stream
) {
    int total = batch * seq_len * embedding_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    embedding_gather_kernel<<<blocks, threads, 0, stream>>>(
        output, input, weight, batch, seq_len, embedding_dim, num_embeddings);
}

void embedding2_forward_launcher(
    float* output,
    const int64_t* input,
    const float* weight,
    int batch,
    int seq_len,
    int embedding_dim,
    int num_embeddings,
    cudaStream_t stream
) {
    int total = batch * seq_len * embedding_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    embedding_gather_kernel<<<blocks, threads, 0, stream>>>(
        output, input, weight, batch, seq_len, embedding_dim, num_embeddings);
}

void add_tensors_launcher(
    float* output,
    const float* a,
    const float* b,
    int total,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    add_tensors_kernel<<<blocks, threads, 0, stream>>>(
        output, a, b, total);
}

} // extern "C"
