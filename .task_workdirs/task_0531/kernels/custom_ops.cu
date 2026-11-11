#include <cuda_runtime.h>

// Embedding lookup kernel: output[i] = weight[indices[i] * embedding_dim + k]
// indices is a flat array of size num_indices, weight is shape [num_embeddings, embedding_dim]
// output is shape [num_indices, embedding_dim], contiguous
__global__ void embedding_kernel(const int64_t* __restrict__ indices,
                                 const float* __restrict__ weight,
                                 float* __restrict__ output,
                                 int64_t num_indices,
                                 int64_t embedding_dim) {
    int64_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = num_indices * embedding_dim;
    if (global_tid >= total_elements) return;

    int64_t index = global_tid / embedding_dim;
    int64_t k = global_tid - index * embedding_dim;
    int64_t weight_row = indices[index];
    output[global_tid] = weight[weight_row * embedding_dim + k];
}

// Clamp minimum kernel: output[i] = max(input[i], min_value)
__global__ void clamp_min_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t numel,
                                float min_value) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; i < numel; i += stride) {
        float val = input[i];
        output[i] = val > min_value ? val : min_value;
    }
}

// Launcher for embedding_forward
extern "C" void embedding_launcher(const int64_t* indices,
                                   const float* weight,
                                   float* output,
                                   int64_t num_indices,
                                   int64_t embedding_dim,
                                   cudaStream_t stream) {
    int64_t total_elements = num_indices * embedding_dim;
    int threads = 256;
    int64_t blocks = (total_elements + threads - 1) / threads;
    // Grid cap check
    if (blocks > 2147483647) {
        blocks = 2147483647;
    }
    embedding_kernel<<<blocks, threads, 0, stream>>>(indices, weight, output, num_indices, embedding_dim);
}

// Launcher for clamp_min_forward
extern "C" void clamp_min_launcher(float* input,
                                  float* output,
                                  int64_t numel,
                                  float min_value,
                                  cudaStream_t stream) {
    int threads = 256;
    int64_t blocks = (numel + threads - 1) / threads;
    if (blocks > 2147483647) {
        blocks = 2147483647;
    }
    clamp_min_kernel<<<blocks, threads, 0, stream>>>(input, output, numel, min_value);
}
