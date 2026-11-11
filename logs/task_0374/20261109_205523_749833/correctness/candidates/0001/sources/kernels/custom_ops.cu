// custom_ops.cu: CUDA kernels for custom_embedding_layernorm and custom_bucketize
#include <cuda_runtime.h>

// Kernel for embedding lookup + LayerNorm
// Each thread processes one token (row of output [B*L, D])
// indices: int64 array of shape [B*L]
// embedding_weight: float [num_embeddings, D]
// gamma, beta: float [D]
// eps: float
// output: float [B*L, D]
__global__ void embedding_layernorm_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ embedding_weight,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float eps,
    int num_embeddings,
    int D,
    int total_tokens,
    float* __restrict__ output
) {
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= total_tokens) return;

    int64_t idx = indices[token_idx];
    // In PyTorch, indices are guaranteed valid after validation in binding
    // But we still clamp for safety (should not happen)
    if (idx < 0) idx = 0;
    if (idx >= num_embeddings) idx = num_embeddings - 1;

    const float* row = embedding_weight + idx * D;
    float* out_row = output + token_idx * D;

    // Compute mean and variance (unbiased=False)
    float mean = 0.0f;
    for (int d = 0; d < D; ++d) {
        mean += row[d];
    }
    mean /= D;

    float var = 0.0f;
    for (int d = 0; d < D; ++d) {
        float diff = row[d] - mean;
        var += diff * diff;
    }
    var /= D;

    float inv_std = rsqrtf(var + eps);

    // Normalize and scale/shift
    for (int d = 0; d < D; ++d) {
        out_row[d] = (row[d] - mean) * inv_std * gamma[d] + beta[d];
    }
}

// Kernel for bucketize: output[i] = first index j such that boundaries[j] > input[i]
// boundaries: sorted ascending float array of length num_boundaries
// Input/output sizes: total_elements
__global__ void bucketize_kernel(
    const float* __restrict__ input,
    const float* __restrict__ boundaries,
    int num_boundaries,
    int total_elements,
    int64_t* __restrict__ output
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_elements) return;

    float val = input[i];
    // Binary search for upper_bound: first index where boundaries[idx] > val
    int low = 0, high = num_boundaries; // high is exclusive
    while (low < high) {
        int mid = (low + high) >> 1;
        if (boundaries[mid] <= val) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    output[i] = (int64_t)low;
}

// Launcher for custom_embedding_layernorm
// Parameters:
//   indices: int64 pointer, shape [B*L]
//   embedding_weight: float pointer, shape [num_embeddings, D]
//   gamma, beta: float pointers, shape [D]
//   eps: float
//   num_embeddings: int
//   embedding_dim: int
//   total_tokens: int (B*L)
//   output: float pointer, shape [B*L, D]
//   stream: cudaStream_t
extern "C" void custom_embedding_layernorm_launcher(
    const int64_t* indices,
    const float* embedding_weight,
    const float* gamma,
    const float* beta,
    float eps,
    int num_embeddings,
    int embedding_dim,
    int total_tokens,
    float* output,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total_tokens + threads - 1) / threads;
    // Optional grid-stride loop not needed if total_tokens < max blocks but launch as is.
    embedding_layernorm_kernel<<<blocks, threads, 0, stream>>>(
        indices, embedding_weight, gamma, beta, eps,
        num_embeddings, embedding_dim, total_tokens, output
    );
}

// Launcher for custom_bucketize
// Parameters:
//   input_values: float pointer, shape [total_elements]
//   boundaries: float pointer, shape [num_boundaries]
//   num_boundaries: int
//   total_elements: int
//   output: int64 pointer, shape [total_elements]
//   stream: cudaStream_t
extern "C" void custom_bucketize_launcher(
    const float* input_values,
    const float* boundaries,
    int num_boundaries,
    int total_elements,
    int64_t* output,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    bucketize_kernel<<<blocks, threads, 0, stream>>>(
        input_values, boundaries, num_boundaries, total_elements, output
    );
}
