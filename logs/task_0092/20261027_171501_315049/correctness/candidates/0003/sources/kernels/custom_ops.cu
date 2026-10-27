#include <cuda_runtime.h>
#include <stdint.h>

// CUDA kernels for custom ops: softmax over dim 1, sort over dim 1, split over dim 1
// All launchers are extern "C" for binding layer in .cpp (not included here)

// Helper to compute linear index given 4D indexing with strides
__device__ __forceinline__ int64_t linear_index(int64_t b, int64_t c, int64_t h, int64_t w,
                                               int64_t stride_b, int64_t stride_c, int64_t stride_h, int64_t stride_w) {
    return b * stride_b + c * stride_c + h * stride_h + w * stride_w;
}

// Softmax kernel: each thread computes softmax over class dimension for one (b,h,w) position.
// Input is tensor of shape [B, C, H, W], dim=1 (C dimension).
// x and out are assumed to be stored with given strides, contiguous or not.
// This kernel is launched with 1D block of threads, each block covers one or more (b,h,w) positions using grid-stride loop.
__global__ void softmax_dim1_kernel(const float* __restrict__ x,
                                    float* __restrict__ out,
                                    int B, int C, int H, int W,
                                    int64_t stride_b, int64_t stride_c, int64_t stride_h, int64_t stride_w) {
    int64_t total_positions = (int64_t)B * H * W;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t pos = idx; pos < total_positions; pos += stride) {
        int64_t b = pos / (int64_t)(H * W);
        int64_t rem = pos - b * (int64_t)(H * W);
        int64_t h = rem / W;
        int64_t w = rem - h * W;

        // Find max for numerical stability
        float max_val = -3.402823466e+38F;
        for (int c = 0; c < C; ++c) {
            int64_t index = linear_index(b, c, h, w, stride_b, stride_c, stride_h, stride_w);
            float val = x[index];
            if (val > max_val) max_val = val;
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            int64_t index = linear_index(b, c, h, w, stride_b, stride_c, stride_h, stride_w);
            float exp_val = __expf(x[index] - max_val);
            out[index] = exp_val;
            sum += exp_val;
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < C; ++c) {
            int64_t index = linear_index(b, c, h, w, stride_b, stride_c, stride_h, stride_w);
            out[index] *= inv_sum;
        }
    }
}

// Bitonic sort kernel for each (b,h,w) position along C dimension.
// Sorts values in place (writing to out_values which is separate from input).
// Also outputs original indices into out_indices as int64_t.
// Since C=10 (fixed in this task), we can hardcode C as a template parameter to avoid dynamic loops.
// But launcher may get arbitrary C, so we provide template with C known at compile time via specialization.
// We'll implement a template with parameter C_VAL, and also a general (non-template) fallback for other C.
// For performance, C=10 is specialized.

template<int C_VAL>
__global__ void bitonic_sort_kernel_dim1(const float* __restrict__ x,
                                         float* __restrict__ out_values,
                                         int64_t* __restrict__ out_indices,
                                         int B, int H, int W,
                                         int64_t stride_b, int64_t stride_c, int64_t stride_h, int64_t stride_w) {
    int64_t total_positions = (int64_t)B * H * W;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t pos = idx; pos < total_positions; pos += stride) {
        int64_t b = pos / (int64_t)(H * W);
        int64_t rem = pos - b * (int64_t)(H * W);
        int64_t h = rem / W;
        int64_t w = rem - h * W;

        float vals[C_VAL];
        int indices[C_VAL];
        // Load and initialize indices
        #pragma unroll
        for (int c = 0; c < C_VAL; ++c) {
            int64_t index = linear_index(b, c, h, w, stride_b, stride_c, stride_h, stride_w);
            vals[c] = x[index];
            indices[c] = c;
        }

        // Bitonic sort: works for any C up to some power of 2; for C not power of 2, pad with +inf
        // But we can use a simpler insertion sort for small C or use bitonic sort with next power of two
        // C=10, next power of two is 16. We'll implement bitonic sort with array size 16, padded with +inf
        // and indices with large dummy.
        const int SORT_N = 16; // next power of two >= C_VAL (for C_VAL=10)
        float sort_vals[SORT_N];
        int sort_indices[SORT_N];
        #pragma unroll
        for (int i = 0; i < SORT_N; ++i) {
            if (i < C_VAL) {
                sort_vals[i] = vals[i];
                sort_indices[i] = indices[i];
            } else {
                sort_vals[i] = 3.402823466e+38F; // +inf
                sort_indices[i] = C_VAL; // dummy
            }
        }

        // Bitonic sort ascending
        #pragma unroll
        for (int k = 2; k <= SORT_N; k <<= 1) {
            #pragma unroll
            for (int j = k >> 1; j > 0; j >>= 1) {
                #pragma unroll
                for (int i = 0; i < SORT_N; ++i) {
                    int l = i ^ j;
                    if (l > i) {
                        bool ascending = ((i & k) == 0);
                        if ((sort_vals[i] > sort_vals[l]) == ascending) {
                            // swap values and indices
                            float tmpv = sort_vals[i];
                            sort_vals[i] = sort_vals[l];
                            sort_vals[l] = tmpv;
                            int tmpi = sort_indices[i];
                            sort_indices[i] = sort_indices[l];
                            sort_indices[l] = tmpi;
                        }
                    }
                }
            }
        }

        // Write out first C_VAL sorted values and indices
        #pragma unroll
        for (int c = 0; c < C_VAL; ++c) {
            int64_t out_index = linear_index(b, c, h, w, stride_b, stride_c, stride_h, stride_w);
            out_values[out_index] = sort_vals[c];
            out_indices[out_index] = (int64_t)sort_indices[c];
        }
    }
}

// General version for arbitrary C (not used for C=10 but provided for safety)
__global__ void bitonic_sort_general_kernel(const float* __restrict__ x,
                                            float* __restrict__ out_values,
                                            int64_t* __restrict__ out_indices,
                                            int B, int C, int H, int W,
                                            int64_t stride_b, int64_t stride_c, int64_t stride_h, int64_t stride_w) {
    int64_t total_positions = (int64_t)B * H * W;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t pos = idx; pos < total_positions; pos += stride) {
        int64_t b = pos / (int64_t)(H * W);
        int64_t rem = pos - b * (int64_t)(H * W);
        int64_t h = rem / W;
        int64_t w = rem - h * W;

        // simple insertion sort for small C
        float vals[128]; // assume C <= 128, but check
        int indices[128];
        if (C > 128) {
            // fallback: not supported, just copy
            for (int c = 0; c < C; ++c) {
                int64_t index = linear_index(b, c, h, w, stride_b, stride_c, stride_h, stride_w);
                out_values[index] = x[index];
                out_indices[index] = c;
            }
            continue;
        }
        for (int c = 0; c < C; ++c) {
            int64_t index = linear_index(b, c, h, w, stride_b, stride_c, stride_h, stride_w);
            vals[c] = x[index];
            indices[c] = c;
        }
        // insertion sort
        for (int i = 1; i < C; ++i) {
            float key_val = vals[i];
            int key_idx = indices[i];
            int j = i - 1;
            while (j >= 0 && vals[j] > key_val) {
                vals[j + 1] = vals[j];
                indices[j + 1] = indices[j];
                j--;
            }
            vals[j + 1] = key_val;
            indices[j + 1] = key_idx;
        }
        for (int c = 0; c < C; ++c) {
            int64_t out_index = linear_index(b, c, h, w, stride_b, stride_c, stride_h, stride_w);
            out_values[out_index] = vals[c];
            out_indices[out_index] = (int64_t)indices[c];
        }
    }
}

// Split copy kernel: output_y is a tensor of shape [B, chunk_size, H, W] representing a slice
// along dim=1 starting at class offset. We copy from input x (shape [B, C, H, W]) the corresponding classes.
// This kernel is launched for one chunk, with offset specifying starting class.
__global__ void split_copy_kernel(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int B, int C, int H, int W,
                                 int64_t stride_b_in, int64_t stride_c_in, int64_t stride_h_in, int64_t stride_w_in,
                                 int offset, int chunk_size,
                                 int64_t stride_b_out, int64_t stride_c_out, int64_t stride_h_out, int64_t stride_w_out) {
    int64_t total = (int64_t)B * chunk_size * H * W;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t pos = idx; pos < total; pos += stride) {
        // Decompose output linear index into b, c_out, h, w
        int64_t tmp = pos;
        int64_t w = tmp % W;
        tmp /= W;
        int64_t h = tmp % H;
        tmp /= H;
        int64_t c_out = tmp % chunk_size;
        int64_t b = tmp / chunk_size;

        int c = offset + c_out;
        if (c < C) {
            int64_t in_index = linear_index(b, c, h, w, stride_b_in, stride_c_in, stride_h_in, stride_w_in);
            int64_t out_index = linear_index(b, c_out, h, w, stride_b_out, stride_c_out, stride_h_out, stride_w_out);
            y[out_index] = x[in_index];
        }
    }
}

// Launcher functions (extern "C" to avoid name mangling, called from binding .cpp)

extern "C" void custom_softmax_forward_launcher(const float* x, float* out,
                                                int B, int C, int H, int W,
                                                int64_t stride_b, int64_t stride_c, int64_t stride_h, int64_t stride_w,
                                                cudaStream_t stream) {
    int threads = 256;
    int64_t total = (int64_t)B * H * W;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 1048576) blocks = 1048576; // grid limit, but grid-stride handles rest
    softmax_dim1_kernel<<<blocks, threads, 0, stream>>>(x, out, B, C, H, W, stride_b, stride_c, stride_h, stride_w);
}

extern "C" void custom_sort_forward_launcher(const float* x, float* out_values, int64_t* out_indices,
                                             int B, int C, int H, int W,
                                             int64_t stride_b, int64_t stride_c, int64_t stride_h, int64_t stride_w,
                                             cudaStream_t stream) {
    int threads = 256;
    int64_t total = (int64_t)B * H * W;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 1048576) blocks = 1048576;

    if (C == 10) {
        bitonic_sort_kernel_dim1<10><<<blocks, threads, 0, stream>>>(x, out_values, out_indices, B, H, W, stride_b, stride_c, stride_h, stride_w);
    } else {
        // fallback general kernel (handles small C via insertion sort)
        bitonic_sort_general_kernel<<<blocks, threads, 0, stream>>>(x, out_values, out_indices, B, C, H, W, stride_b, stride_c, stride_h, stride_w);
    }
}

extern "C" void custom_split_forward_launcher(const float* x, float* out_chunk,
                                              int B, int C, int H, int W,
                                              int64_t stride_b_in, int64_t stride_c_in, int64_t stride_h_in, int64_t stride_w_in,
                                              int offset, int chunk_size,
                                              int64_t stride_b_out, int64_t stride_c_out, int64_t stride_h_out, int64_t stride_w_out,
                                              cudaStream_t stream) {
    int threads = 256;
    int64_t total = (int64_t)B * chunk_size * H * W;
    if (total <= 0) return;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks < 1) blocks = 1;
    if (blocks > 1048576) blocks = 1048576;
    split_copy_kernel<<<blocks, threads, 0, stream>>>(x, out_chunk, B, C, H, W,
                                                      stride_b_in, stride_c_in, stride_h_in, stride_w_in,
                                                      offset, chunk_size,
                                                      stride_b_out, stride_c_out, stride_h_out, stride_w_out);
}
