#include <cuda_runtime.h>

// Embedding forward: gather rows of weight according to indices.
// indices: int64 tensor of shape [batch, seq] (contiguous)
// weight: float32 tensor of shape [num_embeddings, embedding_dim] (contiguous)
// output: float32 tensor of shape [batch, seq, embedding_dim]
// total_indices = batch * seq, embedding_dim = 300
__global__ void embedding_forward_kernel(const int64_t* __restrict__ indices,
                                         const float* __restrict__ weight,
                                         float* __restrict__ output,
                                         int total_indices,
                                         int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_indices; i += stride) {
        int64_t token = indices[i];
        const float* src = weight + token * embedding_dim;
        float* dst = output + (size_t)i * embedding_dim;
        for (int d = 0; d < embedding_dim; ++d) {
            dst[d] = src[d];
        }
    }
}

// Layer normalization over the last dimension (normalized_shape=300) without affine.
// input: float32 tensor of shape [batch, seq, embedding_dim] (contiguous)
// output: same shape
// total_rows = batch * seq, dim = embedding_dim, eps = 1e-5
__global__ void layer_norm_forward_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int total_rows,
                                          int dim,
                                          float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= total_rows) return;

    const float* src = input + (size_t)row * dim;
    float* dst = output + (size_t)row * dim;

    // Compute mean
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) {
        sum += src[i];
    }
    float mean = sum / dim;

    // Compute variance (biased, divide by dim)
    float var = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = src[i] - mean;
        var += diff * diff;
    }
    var /= dim;

    float inv_std = 1.0f / sqrtf(var + eps);

    // Normalize
    for (int i = 0; i < dim; ++i) {
        dst[i] = (src[i] - mean) * inv_std;
    }
}

extern "C" void embedding_forward_launcher(const int64_t* indices,
                                           const float* weight,
                                           float* output,
                                           int total_indices,
                                           int embedding_dim,
                                           cudaStream_t stream) {
    int threads = 256;
    int blocks = (total_indices + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid caps
    embedding_forward_kernel<<<blocks, threads, 0, stream>>>(
        indices, weight, output, total_indices, embedding_dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real binding, this would be checked and throw. Here we just ignore.
        // Binding will check after call via cudaDeviceSynchronize or cudaGetLastError.
    }
}

extern "C" void layer_norm_forward_launcher(const float* input,
                                            float* output,
                                            int total_rows,
                                            int dim,
                                            float eps,
                                            cudaStream_t stream) {
    int threads = 256;
    int blocks = (total_rows + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    layer_norm_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, output, total_rows, dim, eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Ignore here; binding should check.
    }
}
