#include <cuda_runtime.h>
#include <cstdint>

// Softmax over dim=1 for tensor shape [N, C, H, W] where C=10, float32
// Block size = C (10), one block per (n, h, w) position.
// Grid dimension: N * H * W blocks.
// Input/output must be contiguous with strides [C*H*W, H*W, W, 1] (row-major).
// Launcher handles non-contiguous inputs by copying to contiguous in binding.
__global__ void softmax_dim1_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int C, int HW) {
    // blockIdx.x encodes n*h*w position (flattened)
    int pos = blockIdx.x;
    int base = pos * C;
    __shared__ float smem[10];
    // Load C values into shared memory, compute max
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float val = input[base + i];
        smem[i] = val;
        max_val = fmaxf(max_val, val);
    }
    __syncthreads();
    // Reduce max across block (C small, use simple loop)
    // Since C=10 and threads=10, each thread has one element, no need for reduction,
    // but we keep generic: compute shared max via warp shuffle? Simpler: recompute max from smem.
    // Actually after __syncthreads, all smem filled. Each thread can compute max from smem sequentially.
    // But we already computed max_val per thread, need global max. For C=10, use block reduction:
    // Use shared memory to store partial maxes (optional). Since C=10, just recompute max from smem for each thread.
    float block_max = -1e30f;
    for (int i = 0; i < C; ++i) {
        block_max = fmaxf(block_max, smem[i]);
    }
    __syncthreads();
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        smem[i] = __expf(smem[i] - block_max);
    }
    __syncthreads();
    for (int i = 0; i < C; ++i) {
        sum_exp += smem[i];
    }
    __syncthreads();
    // Write output
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        output[base + i] = smem[i] / sum_exp;
    }
}

// Stable sort along dim=1 for C=10 elements, one block per (n, h, w).
// Inputs: values (float), output values and indices (int64).
// Uses shared memory and simple insertion sort (stable).
__global__ void sort_dim1_kernel(const float* __restrict__ input,
                                  float* __restrict__ out_values,
                                  int64_t* __restrict__ out_indices,
                                  int C, int HW) {
    int pos = blockIdx.x;
    int base = pos * C;
    __shared__ float vals[10];
    __shared__ int64_t idx[10];
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        vals[i] = input[base + i];
        idx[i] = i;
    }
    __syncthreads();
    // Insertion sort (stable) executed redundantly by all threads or just thread 0?
    // Since C=10, let each thread do the sort? That would duplicate work but okay.
    // Better: have all threads cooperate on sorting? Simpler: each thread performs the entire sort
    // because C is small, then writes its own element.
    if (threadIdx.x < C) {
        for (int i = 1; i < C; ++i) {
            float key_val = vals[i];
            int64_t key_idx = idx[i];
            int j = i - 1;
            while (j >= 0 && vals[j] > key_val) {
                vals[j + 1] = vals[j];
                idx[j + 1] = idx[j];
                --j;
            }
            vals[j + 1] = key_val;
            idx[j + 1] = key_idx;
        }
    }
    __syncthreads();
    int i = threadIdx.x;
    if (i < C) {
        out_values[base + i] = vals[i];
        out_indices[base + i] = idx[i];
    }
}

// Split: copy subtensors along dim=1 (C) into output tensors.
// Input: contiguous float tensor of shape [N, C, H, W].
// Output: array of output pointers, each with shape [N, chunk_size, H, W], stored contiguously.
// Since binding will allocate each output tensor with shape [N, chunk_size, H, W], we simply copy the corresponding slices.
// The launcher receives output pointers for all chunks.
__global__ void split_copy_kernel(const float* __restrict__ input,
                                  float** __restrict__ outputs,
                                  int C, int chunk_size, int num_chunks, int HW) {
    // Grid: one block per (n, h*w) -> total N*HW blocks.
    int pos = blockIdx.x; // n*HW + hw
    int n = pos / HW;
    int hw = pos % HW;
    int input_offset = n * C * HW + hw;
    int chunk_idx = threadIdx.x; // chunk index (0..num_chunks-1)
    if (chunk_idx >= num_chunks) return;
    int this_chunk_size = (chunk_idx < num_chunks - 1) ? chunk_size : (C - chunk_idx * chunk_size);
    if (this_chunk_size <= 0) return;
    const float* src = input + input_offset + chunk_idx * chunk_size * HW;
    float* dst = outputs[chunk_idx] + n * this_chunk_size * HW + hw;
    // copy this_chunk_size elements along dim=1
    for (int c = 0; c < this_chunk_size; ++c) {
        dst[c * HW] = src[c * HW];
    }
}

// Launcher functions
extern "C" void softmax_forward_launcher(const float* input,
                                           float* output,
                                           int N, int C, int H, int W,
                                           cudaStream_t stream) {
    int HW = H * W;
    int num_blocks = N * HW;
    softmax_dim1_kernel<<<num_blocks, C, 0, stream>>>(input, output, C, HW);
}

extern "C" void sort_forward_launcher(const float* input,
                                        float* out_values,
                                        int64_t* out_indices,
                                        int N, int C, int H, int W,
                                        cudaStream_t stream) {
    int HW = H * W;
    int num_blocks = N * HW;
    sort_dim1_kernel<<<num_blocks, C, 0, stream>>>(input, out_values, out_indices, C, HW);
}

extern "C" void split_forward_launcher(const float* input,
                                         float** outputs,
                                         int N, int C, int H, int W,
                                         int split_size, int num_chunks,
                                         cudaStream_t stream) {
    int HW = H * W;
    int num_blocks = N * HW;
    // Each block handles one (n,hw) position and all chunks by looping over chunks in threads.
    // We launch with num_chunks threads per block.
    // But kernel assumes threadIdx.x corresponds to chunk idx, so blockDim.x = num_chunks.
    split_copy_kernel<<<num_blocks, num_chunks, 0, stream>>>(input, outputs, C, split_size, num_chunks, HW);
}
