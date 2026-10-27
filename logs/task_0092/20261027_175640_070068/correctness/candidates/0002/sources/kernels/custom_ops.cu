#include <cuda_runtime.h>
#include <cstdint>

// Helper to compute linear index from 4D strides
__device__ __forceinline__ int64_t idx4(int64_t n, int64_t c, int64_t h, int64_t w,
                                        int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w) {
    return n * stride_n + c * stride_c + h * stride_h + w * stride_w;
}

// Softmax along dim=1 (C dimension)
// Input: x shape (N, C, H, W), output same shape
// Each thread block handles one (n, h, w) position, loops over C
__global__ void softmax_dim1_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int N, int C, int H, int W,
                                     int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w) {
    int64_t total_positions = (int64_t)N * H * W;
    int64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total_positions) return;

    int n = pos / (H * W);
    int rem = pos % (H * W);
    int h = rem / W;
    int w = rem % W;

    // Find max for numerical stability
    float max_val = -1e30f;
    for (int c = 0; c < C; ++c) {
        int64_t idx = idx4(n, c, h, w, stride_n, stride_c, stride_h, stride_w);
        float v = input[idx];
        if (v > max_val) max_val = v;
    }

    // Compute exp sum
    float sum_exp = 0.0f;
    for (int c = 0; c < C; ++c) {
        int64_t idx = idx4(n, c, h, w, stride_n, stride_c, stride_h, stride_w);
        float v = expf(input[idx] - max_val);
        output[idx] = v;
        sum_exp += v;
    }

    // Normalize
    float inv_sum = 1.0f / sum_exp;
    for (int c = 0; c < C; ++c) {
        int64_t idx = idx4(n, c, h, w, stride_n, stride_c, stride_h, stride_w);
        output[idx] *= inv_sum;
    }
}

// Sort along dim=1 (C dimension) using insertion sort within one thread per (n,h,w)
// For small C (like 10), insertion sort is efficient and stable.
// Two kernels to optionally produce indices.
__global__ void sort_dim1_values_kernel(const float* __restrict__ input,
                                         float* __restrict__ output,
                                         int N, int C, int H, int W,
                                         int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w) {
    int64_t total_positions = (int64_t)N * H * W;
    int64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total_positions) return;

    int n = pos / (H * W);
    int rem = pos % (H * W);
    int h = rem / W;
    int w = rem % W;

    float vals[32]; // assume C <= 32, currently C=10
    for (int c = 0; c < C; ++c) {
        int64_t idx = idx4(n, c, h, w, stride_n, stride_c, stride_h, stride_w);
        vals[c] = input[idx];
    }

    // Insertion sort (stable)
    for (int i = 1; i < C; ++i) {
        float key = vals[i];
        int j = i - 1;
        while (j >= 0 && vals[j] > key) {
            vals[j + 1] = vals[j];
            j--;
        }
        vals[j + 1] = key;
    }

    for (int c = 0; c < C; ++c) {
        int64_t idx = idx4(n, c, h, w, stride_n, stride_c, stride_h, stride_w);
        output[idx] = vals[c];
    }
}

__global__ void sort_dim1_values_indices_kernel(const float* __restrict__ input,
                                                 float* __restrict__ sorted_values,
                                                 int64_t* __restrict__ indices,
                                                 int N, int C, int H, int W,
                                                 int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w) {
    int64_t total_positions = (int64_t)N * H * W;
    int64_t pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= total_positions) return;

    int n = pos / (H * W);
    int rem = pos % (H * W);
    int h = rem / W;
    int w = rem % W;

    float vals[32];
    int64_t idxs[32];
    for (int c = 0; c < C; ++c) {
        int64_t idx = idx4(n, c, h, w, stride_n, stride_c, stride_h, stride_w);
        vals[c] = input[idx];
        idxs[c] = c;
    }

    // Insertion sort (stable by only swapping when strictly greater)
    for (int i = 1; i < C; ++i) {
        float key_val = vals[i];
        int64_t key_idx = idxs[i];
        int j = i - 1;
        while (j >= 0 && vals[j] > key_val) {
            vals[j + 1] = vals[j];
            idxs[j + 1] = idxs[j];
            j--;
        }
        vals[j + 1] = key_val;
        idxs[j + 1] = key_idx;
    }

    for (int c = 0; c < C; ++c) {
        int64_t idx = idx4(n, c, h, w, stride_n, stride_c, stride_h, stride_w);
        sorted_values[idx] = vals[c];
        indices[idx] = idxs[c];
    }
}

// Launchers
extern "C" void custom_softmax_dim1_launcher(const float* input,
                                               float* output,
                                               int64_t N, int64_t C, int64_t H, int64_t W,
                                               int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
                                               cudaStream_t stream) {
    int64_t total_positions = N * H * W;
    int threads = 256;
    int64_t blocks = (total_positions + threads - 1) / threads;
    if (blocks > 0) {
        softmax_dim1_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
            input, output, (int)N, (int)C, (int)H, (int)W,
            stride_n, stride_c, stride_h, stride_w);
    }
}

extern "C" void custom_sort_dim1_launcher(const float* input,
                                           float* sorted_values,
                                           int64_t* indices, // can be nullptr if return_key=false
                                           int return_key,
                                           int64_t N, int64_t C, int64_t H, int64_t W,
                                           int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
                                           cudaStream_t stream) {
    int64_t total_positions = N * H * W;
    int threads = 256;
    int64_t blocks = (total_positions + threads - 1) / threads;
    if (blocks > 0) {
        if (return_key) {
            sort_dim1_values_indices_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
                input, sorted_values, indices, (int)N, (int)C, (int)H, (int)W,
                stride_n, stride_c, stride_h, stride_w);
        } else {
            sort_dim1_values_kernel<<<(unsigned int)blocks, threads, 0, stream>>>(
                input, sorted_values, (int)N, (int)C, (int)H, (int)W,
                stride_n, stride_c, stride_h, stride_w);
        }
    }
}

// Split along dim=1. Since splitting is a view operation and we cannot return tensors,
// this function simply computes the offsets and sizes for each chunk. The binding layer
// will use these to create narrow views. Here we define a helper that can be called from
// C++ (binding) but since this .cu cannot include torch, we expose an extern "C" function
// that fills arrays with chunk offsets.
extern "C" void custom_split_dim1_offsets(int64_t C, int64_t chunk_size,
                                         int64_t* offsets_out, int64_t* sizes_out,
                                         int* num_chunks_out) {
    int num_chunks = (int)((C + chunk_size - 1) / chunk_size);
    *num_chunks_out = num_chunks;
    for (int i = 0; i < num_chunks; ++i) {
        int64_t start = i * chunk_size;
        int64_t size = (start + chunk_size <= C) ? chunk_size : (C - start);
        offsets_out[i] = start;
        sizes_out[i] = size;
    }
}
