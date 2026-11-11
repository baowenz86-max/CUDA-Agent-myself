#include <cuda_runtime.h>

// Embedding forward kernel
// indices: int64_t pointer, may have arbitrary strides specified by caller via linearized mapping
// weight: float pointer, shape [num_embeddings, embedding_dim], contiguous
// output: float pointer, shape [total_indices, embedding_dim], contiguous
// total_indices: number of index elements (product of indices shape)
// embedding_dim: dimension of embedding vectors
__global__ void embedding_forward_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t total_indices,
    int64_t embedding_dim)
{
    int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t flat = global_idx; flat < total_indices; flat += stride) {
        int64_t idx = indices[flat];
        // PyTorch embedding treats out-of-range indices as undefined, but contract says valid indices.
        // We skip safety to match contract; caller ensures indices in range.
        const float* src = weight + idx * embedding_dim;
        float* dst = output + flat * embedding_dim;
        for (int64_t j = 0; j < embedding_dim; ++j) {
            dst[j] = src[j];
        }
    }
}

// Launcher for embedding forward. Accepts raw pointers and stream.
// The binding layer is responsible for ensuring tensors are on CUDA and appropriate dtypes.
// It should pass contiguous indices/weight/output pointers. The binding layer may call
// .contiguous() if needed.
extern "C" void embedding_forward_launcher(
    const int64_t* indices,
    const float* weight,
    float* output,
    int64_t total_indices,
    int64_t embedding_dim,
    cudaStream_t stream)
{
    if (total_indices <= 0 || embedding_dim <= 0) {
        return; // nothing to do for empty tensors
    }

    const int threads = 256;
    int64_t blocks64 = (total_indices + threads - 1) / threads;
    // Cap blocks to avoid excessive grid size for small inputs
    int blocks = (int)min(blocks64, (int64_t)65535);

    embedding_forward_kernel<<<blocks, threads, 0, stream>>>(
        indices, weight, output, total_indices, embedding_dim);
}
