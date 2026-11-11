#include <cuda_runtime.h>

// Embedding forward kernel: output shape [N0, N1, embedding_dim]
// indices: int64 tensor shape [N0, N1] (may have arbitrary strides)
// weight: float32 tensor shape [num_embeddings, embedding_dim] contiguous
// output: float32 contiguous [N0, N1, embedding_dim]
__global__ void embedding_forward_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t N0,
    int64_t N1,
    int64_t embedding_dim,
    int64_t num_embeddings) {
    int64_t total = N0 * N1 * embedding_dim;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        // decompose flat index into (b, j, k)
        int64_t k = idx % embedding_dim;
        int64_t tmp = idx / embedding_dim;
        int64_t j = tmp % N1;
        int64_t b = tmp / N1;
        
        // indices may be non-contiguous, but we treat it as contiguous in the kernel,
        // assuming the binding layer ensures contiguity or the strides are handled by caller.
        // Here, we simply use the flat index for indices tensor as (b * N1 + j).
        // This is correct if indices is contiguous. If not, binding layer should make it contiguous.
        int64_t index = indices[b * N1 + j];
        // PyTorch embedding out-of-bounds behavior: undefined; we just read without bounds check to match.
        int idx_weight = index * embedding_dim + k;
        output[idx] = weight[idx_weight];
    }
}

// Mean over dim=1 kernel: input shape [N0, N1, N2], output [N0, N2]
// x: float32 contiguous (or we assume contiguous; binding layer ensures)
// output: float32 contiguous
__global__ void mean_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ output,
    int64_t N0,
    int64_t N1,
    int64_t N2) {
    int64_t total = N0 * N2;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int64_t d = idx % N2;
        int64_t b = idx / N2;
        float sum = 0.0f;
        for (int64_t j = 0; j < N1; ++j) {
            sum += x[(b * N1 + j) * N2 + d];
        }
        output[idx] = sum / (float)N1;
    }
}

extern "C" void embedding_forward_launcher(
    const int64_t* indices,
    const float* weight,
    float* output,
    int64_t N0,
    int64_t N1,
    int64_t embedding_dim,
    int64_t num_embeddings,
    cudaStream_t stream) {
    int64_t total = N0 * N1 * embedding_dim;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid limit; kernel uses grid-stride loop
    embedding_forward_kernel<<<blocks, threads, 0, stream>>>(
        indices, weight, output, N0, N1, embedding_dim, num_embeddings);
}

extern "C" void mean_dim1_launcher(
    const float* x,
    float* output,
    int64_t N0,
    int64_t N1,
    int64_t N2,
    cudaStream_t stream) {
    int64_t total = N0 * N2;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    mean_dim1_kernel<<<blocks, threads, 0, stream>>>(x, output, N0, N1, N2);
}
