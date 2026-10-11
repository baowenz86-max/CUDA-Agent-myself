#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include "attention_launchers.h"

__global__ void pack_kernel(const float* projection, const float* bias, float* packed,
                            int batch, int sequence, int embed, int heads, bool batch_first) {
    int depth = embed / heads;
    int64_t count = static_cast<int64_t>(batch) * sequence * embed;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int channel = index % depth;
        int token = (index / depth) % sequence;
        int head = (index / (static_cast<int64_t>(depth) * sequence)) % heads;
        int sample = index / (static_cast<int64_t>(sequence) * embed);
        int64_t row = batch_first ? static_cast<int64_t>(sample) * sequence + token
                                 : static_cast<int64_t>(token) * batch + sample;
        int feature = head * depth + channel;
        packed[index] = projection[row * embed + feature] + bias[feature];
    }
}

__global__ void softmax_kernel(float* scores, int64_t rows, int keys) {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    for (int64_t row = static_cast<int64_t>(blockIdx.x) * 8 + warp;
         row < rows; row += static_cast<int64_t>(gridDim.x) * 8) {
        float* values = scores + row * keys;
        float maximum = -CUDART_INF_F;
        for (int64_t column = lane; column < keys; column += 32)
            maximum = fmaxf(maximum, values[column]);
        for (int offset = 16; offset > 0; offset /= 2)
            maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
        maximum = __shfl_sync(0xffffffff, maximum, 0);
        float total = 0.0f;
        for (int64_t column = lane; column < keys; column += 32) {
            float probability = expf(values[column] - maximum);
            values[column] = probability;
            total += probability;
        }
        for (int offset = 16; offset > 0; offset /= 2)
            total += __shfl_down_sync(0xffffffff, total, offset);
        float inverse = 1.0f / __shfl_sync(0xffffffff, total, 0);
        for (int64_t column = lane; column < keys; column += 32)
            values[column] *= inverse;
    }
}

__global__ void merge_kernel(const float* context, float* merged, int batch,
                             int sequence, int embed, int heads, bool batch_first) {
    int depth = embed / heads;
    int64_t count = static_cast<int64_t>(batch) * sequence * embed;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int feature = index % embed;
        int64_t row = index / embed;
        int sample = batch_first ? row / sequence : row % batch;
        int token = batch_first ? row % sequence : row / batch;
        int64_t source = ((static_cast<int64_t>(sample) * heads + feature / depth) * sequence + token) * depth + feature % depth;
        merged[index] = context[source];
    }
}

__global__ void epilogue_kernel(const float* projected, const float* bias, float* output,
                                int batch, int sequence, int embed, int repeats,
                                float scale, bool batch_first) {
    int64_t count = static_cast<int64_t>(batch) * sequence * embed * repeats;
    int64_t repeated_axis = static_cast<int64_t>(batch_first ? sequence : batch) * repeats;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        int feature = index % embed;
        int64_t row = index / embed;
        int64_t outer = row / repeated_axis;
        int64_t inner = (row % repeated_axis) / repeats;
        int64_t source_row = outer * (batch_first ? sequence : batch) + inner;
        output[index] = tanf((projected[source_row * embed + feature] + bias[feature]) * scale);
    }
}

static int blocks_for(int64_t count) {
    int64_t blocks = (count + 255) / 256;
    return static_cast<int>(blocks > 65535 ? 65535 : blocks);
}

void attention_pack(const float* projection, const float* bias, float* packed,
                    int batch, int sequence, int embed, int heads, bool batch_first, cudaStream_t stream) {
    pack_kernel<<<blocks_for(static_cast<int64_t>(batch) * sequence * embed), 256, 0, stream>>>(
        projection, bias, packed, batch, sequence, embed, heads, batch_first);
}

void attention_softmax(float* scores, int64_t rows, int keys, cudaStream_t stream) {
    int64_t needed = (rows + 7) / 8;
    int blocks = static_cast<int>(needed > 65535 ? 65535 : needed);
    softmax_kernel<<<blocks, 256, 0, stream>>>(scores, rows, keys);
}

void attention_merge(const float* context, float* merged, int batch, int sequence,
                     int embed, int heads, bool batch_first, cudaStream_t stream) {
    merge_kernel<<<blocks_for(static_cast<int64_t>(batch) * sequence * embed), 256, 0, stream>>>(
        context, merged, batch, sequence, embed, heads, batch_first);
}

void attention_epilogue(const float* projected, const float* bias, float* output,
                        int batch, int sequence, int embed, int repeats,
                        float scale, bool batch_first, cudaStream_t stream) {
    epilogue_kernel<<<blocks_for(static_cast<int64_t>(batch) * sequence * embed * repeats), 256, 0, stream>>>(
        projected, bias, output, batch, sequence, embed, repeats, scale, batch_first);
}
