#include <cuda_runtime.h>
#include <stdint.h>

// Helper: get index from int64 tensor at flattened position
__device__ __forceinline__ int64_t get_index(const int64_t* data, int64_t idx) {
    return data[idx];
}

// Kernel: embedding lookup
// weight: [num_embeddings, embedding_dim] float32
// input_ids: [batch_size, seq_len] int64
// output: [batch_size, seq_len, embedding_dim] float32
// vectorized by float4, each thread handles 4 consecutive embedding dims
__global__ void embedding_lookup_kernel(
    const float* __restrict__ weight,
    const int64_t* __restrict__ input_ids,
    float* __restrict__ output,
    int64_t num_embeddings,
    int64_t embedding_dim,
    int64_t batch_size,
    int64_t seq_len) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch_size * seq_len;
    if (idx >= total) return;

    int64_t b = idx / seq_len;
    int64_t s = idx - b * seq_len;
    int64_t token = input_ids[idx];
    
    // Bounds check for safety
    if (token < 0 || token >= num_embeddings) {
        // write zeros to all embedding_dim (vectorized)
        int64_t out_offset = (b * seq_len + s) * embedding_dim;
        int64_t vec_count = embedding_dim / 4;
        float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4* out4 = reinterpret_cast<float4*>(output + out_offset);
        for (int64_t v = 0; v < vec_count; ++v) {
            out4[v] = zero;
        }
        // handle remainder if embedding_dim not divisible by 4 (though here it is)
        int64_t rem = embedding_dim - vec_count * 4;
        if (rem > 0) {
            int64_t rem_offset = out_offset + vec_count * 4;
            for (int64_t r = 0; r < rem; ++r) {
                output[rem_offset + r] = 0.0f;
            }
        }
        return;
    }

    const float* weight_row = weight + token * embedding_dim;
    float* out_row = output + (b * seq_len + s) * embedding_dim;
    
    // vectorized copy using float4
    int64_t vec_count = embedding_dim / 4;
    const float4* weight_row4 = reinterpret_cast<const float4*>(weight_row);
    float4* out_row4 = reinterpret_cast<float4*>(out_row);
    for (int64_t v = 0; v < vec_count; ++v) {
        out_row4[v] = weight_row4[v];
    }
    // remainder (should be 0 for embedding_dim=1024)
    int64_t rem = embedding_dim - vec_count * 4;
    if (rem > 0) {
        int64_t rem_offset = vec_count * 4;
        for (int64_t r = 0; r < rem; ++r) {
            out_row[rem_offset + r] = weight_row[rem_offset + r];
        }
    }
}

// Kernel: index_select along specific dim (currently supports dim=1 for 3D input)
// input: [batch_size, src_len, embedding_dim] float32
// index: [index_len] int64
// output: [batch_size, index_len, embedding_dim] float32
// dim must be 1 (but we allow generic dim for 3D? contract only uses dim=1)
__global__ void index_select_kernel_dim1(
    const float* __restrict__ input,
    const int64_t* __restrict__ index,
    float* __restrict__ output,
    int64_t batch_size,
    int64_t src_len,
    int64_t index_len,
    int64_t embedding_dim) {
    
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch_size * index_len;
    if (idx >= total) return;

    int64_t b = idx / index_len;
    int64_t i = idx - b * index_len;
    int64_t src_idx = index[i];
    
    // bounds check
    if (src_idx < 0 || src_idx >= src_len) {
        // write zeros
        int64_t out_offset = (b * index_len + i) * embedding_dim;
        int64_t vec_count = embedding_dim / 4;
        float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4* out4 = reinterpret_cast<float4*>(output + out_offset);
        for (int64_t v = 0; v < vec_count; ++v) {
            out4[v] = zero;
        }
        int64_t rem = embedding_dim - vec_count * 4;
        if (rem > 0) {
            int64_t rem_offset = out_offset + vec_count * 4;
            for (int64_t r = 0; r < rem; ++r) {
                output[rem_offset + r] = 0.0f;
            }
        }
        return;
    }

    const float* src_row = input + (b * src_len + src_idx) * embedding_dim;
    float* out_row = output + (b * index_len + i) * embedding_dim;
    
    int64_t vec_count = embedding_dim / 4;
    const float4* src_row4 = reinterpret_cast<const float4*>(src_row);
    float4* out_row4 = reinterpret_cast<float4*>(out_row);
    for (int64_t v = 0; v < vec_count; ++v) {
        out_row4[v] = src_row4[v];
    }
    int64_t rem = embedding_dim - vec_count * 4;
    if (rem > 0) {
        int64_t rem_offset = vec_count * 4;
        for (int64_t r = 0; r < rem; ++r) {
            out_row[rem_offset + r] = src_row[rem_offset + r];
        }
    }
}

// Launcher for embedding lookup
extern "C" void embedding_lookup_launcher(
    const float* weight,
    const int64_t* input_ids,
    float* output,
    int64_t num_embeddings,
    int64_t embedding_dim,
    int64_t batch_size,
    int64_t seq_len,
    cudaStream_t stream) {
    
    int64_t total = batch_size * seq_len;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    embedding_lookup_kernel<<<blocks, threads, 0, stream>>>(
        weight, input_ids, output, num_embeddings, embedding_dim, batch_size, seq_len);
}

// Launcher for index_select (assumes dim=1, but signature includes dim for future)
extern "C" void index_select_launcher(
    const float* input,
    int64_t dim,
    const int64_t* index,
    float* output,
    int64_t batch_size,
    int64_t src_len,
    int64_t index_len,
    int64_t embedding_dim,
    cudaStream_t stream) {
    
    // Only support dim=1 for this contract; if dim != 1, we could fallback (but binding should ensure dim=1)
    // For robustness, if dim != 1, do nothing or call a generic kernel? Since contract only has dim=1, we can assume.
    int64_t total = batch_size * index_len;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    index_select_kernel_dim1<<<blocks, threads, 0, stream>>>(
        input, index, output, batch_size, src_len, index_len, embedding_dim);
}
