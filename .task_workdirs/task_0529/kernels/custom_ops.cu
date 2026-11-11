#include <cuda_runtime.h>

// embedding_forward kernel: output[b, s, d] = weight[indices[b, s] * embedding_dim + d]
// indices shape: [batch, seq_len] int64
// weight shape: [num_embeddings, embedding_dim] float
// output shape: [batch, seq_len, embedding_dim] float
__global__ void embedding_forward_kernel(
    const float* __restrict__ weight,
    const int64_t* __restrict__ indices,
    float* __restrict__ output,
    int num_embeddings,
    int embedding_dim,
    int batch,
    int seq_len)
{
    int64_t total = (int64_t)batch * seq_len * embedding_dim;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        int d = i % embedding_dim;
        int64_t tmp = i / embedding_dim;
        int s = tmp % seq_len;
        int b = tmp / seq_len;
        int64_t row = indices[(int64_t)b * seq_len + s];
        // 边界检查（调用方保证索引合法，但这里仍做防御）
        if (row < 0 || row >= num_embeddings) {
            output[i] = 0.0f;
        } else {
            output[i] = weight[row * embedding_dim + d];
        }
    }
}

// amax_dim1_keepdim kernel: output[b, 0, d] = max over s of input[b, s, d]
// input shape: [batch, seq_len, embedding_dim] float
// output shape: [batch, 1, embedding_dim] float
// 每个 block 处理一个 (b, d) 对，使用 shared memory 归约
__global__ void amax_dim1_keepdim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int seq_len,
    int embedding_dim)
{
    extern __shared__ float smem[]; // 大小 = blockDim.x

    int b = blockIdx.x;
    int d = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    float max_val = -3.402823466e+38f; // -FLT_MAX
    const float* base = input + ((int64_t)b * seq_len + 0) * embedding_dim + d;
    // 每个线程处理多个 s
    for (int s = tid; s < seq_len; s += block_size) {
        float val = base[(int64_t)s * embedding_dim];
        if (val > max_val) max_val = val;
    }

    smem[tid] = max_val;
    __syncthreads();

    // 归约
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = smem[tid + stride];
            if (other > smem[tid]) smem[tid] = other;
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[(int64_t)b * embedding_dim + d] = smem[0];
    }
}

extern "C" void embedding_forward_launcher(
    const float* weight,
    const int64_t* indices,
    float* output,
    int num_embeddings,
    int embedding_dim,
    int batch,
    int seq_len,
    cudaStream_t stream)
{
    int total_threads = batch * seq_len * embedding_dim;
    int block_size = 256;
    int blocks = (total_threads + block_size - 1) / block_size;
    if (blocks > 65535) {
        blocks = 65535;
    }
    embedding_forward_kernel<<<blocks, block_size, 0, stream>>>(
        weight, indices, output,
        num_embeddings, embedding_dim, batch, seq_len);
}

extern "C" void amax_dim1_keepdim_launcher(
    const float* input,
    float* output,
    int batch,
    int seq_len,
    int embedding_dim,
    cudaStream_t stream)
{
    dim3 blocks(batch, embedding_dim);
    int block_size = 256;
    // 限制 block 大小不超过 seq_len，但 seq_len=1024，256 即可
    if (block_size > seq_len) block_size = seq_len;
    int smem_size = block_size * sizeof(float);
    amax_dim1_keepdim_kernel<<<blocks, block_size, smem_size, stream>>>(
        input, output, batch, seq_len, embedding_dim);
}
