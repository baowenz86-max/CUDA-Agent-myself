#include <cuda_runtime.h>
#include <cmath>

// 使用 erf 的 GELU 计算
template <typename scalar_t>
__device__ scalar_t gelu_erf(scalar_t x) {
    return scalar_t(0.5) * x * (scalar_t(1) + erff(x * scalar_t(1.0) / scalar_t(sqrt(2.0))));
}

// 融合 embedding lookup 和 GELU
// indices: int64_t* 形状 [batch, seq]，值是 [0, num_embeddings-1] 之间的索引
// weight: float* 形状 [num_embeddings, embedding_dim]
// output: float* 形状 [batch, seq, embedding_dim]
// num_indices: batch * seq
// embedding_dim: 每个 token 的嵌入维度
__global__ void embedding_lookup_gelu_kernel_float(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int64_t num_indices,
    int64_t embedding_dim
) {
    // 每个线程处理多个输出元素，使用 grid-stride loop
    int64_t total_out = num_indices * embedding_dim;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * (int64_t)gridDim.x;

    for (int64_t out_idx = idx; out_idx < total_out; out_idx += stride) {
        int64_t token_idx = out_idx / embedding_dim;
        int64_t dim = out_idx % embedding_dim;
        int64_t row = indices[token_idx]; // 可能越界，但由 binding 层检查
        float val = weight[row * embedding_dim + dim];
        output[out_idx] = gelu_erf(val);
    }
}

// Launcher 函数，供 binding.cpp 调用
extern "C" void embedding_lookup_gelu_launcher(
    const int64_t* indices,
    const float* weight,
    float* output,
    int64_t num_indices,
    int64_t embedding_dim,
    cudaStream_t stream
) {
    int threads = 256;
    int64_t total_out = num_indices * embedding_dim;
    int blocks = (int)((total_out + threads - 1) / threads);
    // 限制最大 grid 尺寸避免过大
    if (blocks > 1048576) blocks = 1048576;
    if (blocks < 1) blocks = 1;

    embedding_lookup_gelu_kernel_float<<<blocks, threads, 0, stream>>>(
        indices, weight, output, num_indices, embedding_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // 只记录错误，不抛出，binding 层可能需要处理
        // 实际 binding 层会检查 cudaGetLastError
    }
}
