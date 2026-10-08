#include <cuda_runtime.h>

#include <float.h>

namespace {

constexpr int kThreads = 64;

__global__ void add_bias_kernel(
    float* values, const float* bias, int rows, int columns) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = rows * columns;
    if (index < count) {
        values[index] += bias[index % columns];
    }
}

__global__ void attention_kernel(
    float* query_and_output,
    const float* key,
    const float* value,
    const bool* key_padding_mask,
    int batch_size,
    int sequence_length,
    int embed_dim,
    int num_heads) {
    extern __shared__ float scores[];

    const int head_dim = embed_dim / num_heads;
    const int lane = threadIdx.x;
    const int block = blockIdx.x;
    const int head = block % num_heads;
    const int query_index = (block / num_heads) % sequence_length;
    const int batch = block / (num_heads * sequence_length);

    if (batch >= batch_size || lane >= sequence_length) {
        return;
    }

    const int batch_offset = batch * sequence_length * embed_dim;
    const int query_offset = batch_offset + query_index * embed_dim + head * head_dim;
    const int key_offset = batch_offset + lane * embed_dim + head * head_dim;
    float score = 0.0f;
    #pragma unroll
    for (int dim = 0; dim < 64; ++dim) {
        score = fmaf(query_and_output[query_offset + dim], key[key_offset + dim], score);
    }
    score *= 0.125f;
    if (key_padding_mask[batch * sequence_length + lane]) {
        score = -FLT_MAX;
    }
    scores[lane] = score;
    __syncthreads();

    for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            scores[lane] = fmaxf(scores[lane], scores[lane + offset]);
        }
        __syncthreads();
    }

    const float maximum = scores[0];
    const float probability = expf(score - maximum);
    scores[lane] = probability;
    scores[kThreads + lane] = probability;
    __syncthreads();

    for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
        if (lane < offset) {
            scores[kThreads + lane] += scores[kThreads + lane + offset];
        }
        __syncthreads();
    }

    const float reciprocal_sum = 1.0f / scores[kThreads];
    float result = 0.0f;
    for (int token = 0; token < sequence_length; ++token) {
        const float weight = scores[token] * reciprocal_sum;
        result = fmaf(
            weight,
            value[batch_offset + token * embed_dim + head * head_dim + lane],
            result);
    }
    query_and_output[query_offset + lane] = result;
}

__global__ void copy_and_multiply_kernel(
    float* stacked, const float* value_input, const float* multiplier, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        stacked[index] = value_input[index];
        stacked[count + index] *= multiplier[index];
    }
}

}  // namespace

extern "C" void launch_add_bias(
    float* values, const float* bias, int rows, int columns, cudaStream_t stream) {
    constexpr int threads = 256;
    const int count = rows * columns;
    const int blocks = (count + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(values, bias, rows, columns);
}

extern "C" void launch_attention(
    float* query_and_output,
    const float* key,
    const float* value,
    const bool* key_padding_mask,
    int batch_size,
    int sequence_length,
    int embed_dim,
    int num_heads,
    cudaStream_t stream) {
    const int blocks = batch_size * sequence_length * num_heads;
    attention_kernel<<<blocks, kThreads, 2 * kThreads * sizeof(float), stream>>>(
        query_and_output,
        key,
        value,
        key_padding_mask,
        batch_size,
        sequence_length,
        embed_dim,
        num_heads);
}

extern "C" void launch_copy_and_multiply(
    float* stacked,
    const float* value_input,
    const float* multiplier,
    int count,
    cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    copy_and_multiply_kernel<<<blocks, threads, 0, stream>>>(
        stacked, value_input, multiplier, count);
}
