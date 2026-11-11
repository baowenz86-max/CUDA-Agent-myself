#include <cuda_runtime.h>

// ============================================================================
// custom_softmax: computes softmax along a specified dimension.
// Assumes float32 tensors, arbitrary strides.
// Input: x pointer, dim (0..3), shape[4], strides[4]
// Output: y pointer
// ============================================================================

__global__ void softmax_dim_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int dim,
                                   int outer_size,
                                   int inner_size,
                                   int dim_size,
                                   unsigned int input_dim_stride,
                                   unsigned int output_dim_stride) {
    // Each block handles one outer*inner combination to leverage shared memory reductions.
    __shared__ float s_max, s_sum;
    
    int outer = blockIdx.x;
    int inner = blockIdx.y;
    if (outer >= outer_size || inner >= inner_size) return;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // Base offset for this (outer, inner) slice
    // For dim=1, outer is n, inner is linear h*W+w
    // Physical offset for (outer, d, inner) is outer * stride[0] + d * stride[1] + inner, since stride[2]=128, stride[3]=1, inner = h*128+w
    // But we need to compute using input_dim_stride (stride[1]) and outer stride (stride[0] = outer_stride)
    unsigned int outer_stride = input_dim_stride * dim_size; // assume contiguous preceding dims
    unsigned int base = outer * outer_stride + inner;

    // Compute max
    float max_val = -1e30f;
    for (int d = tid; d < dim_size; d += num_threads) {
        unsigned int idx = base + d * input_dim_stride;
        float val = input[idx];
        if (val > max_val) max_val = val;
    }
    // block reduction for max
    __shared__ float max_buffer[256];
    if (tid < num_threads) max_buffer[tid] = max_val;
    __syncthreads();
    for (int s = num_threads/2; s > 0; s >>= 1) {
        if (tid < s) {
            max_buffer[tid] = fmaxf(max_buffer[tid], max_buffer[tid + s]);
        }
        __syncthreads();
    }
    float block_max = max_buffer[0];
    __syncthreads();

    // Compute sum of exp(x - max)
    float sum_val = 0.0f;
    for (int d = tid; d < dim_size; d += num_threads) {
        unsigned int idx = base + d * input_dim_stride;
        float val = input[idx];
        sum_val += __expf(val - block_max);
    }
    // block reduction for sum
    __shared__ float sum_buffer[256];
    if (tid < num_threads) sum_buffer[tid] = sum_val;
    __syncthreads();
    for (int s = num_threads/2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_buffer[tid] += sum_buffer[tid + s];
        }
        __syncthreads();
    }
    float block_sum = sum_buffer[0];
    __syncthreads();

    // Write output normalized
    for (int d = tid; d < dim_size; d += num_threads) {
        unsigned int idx = base + d * input_dim_stride;
        float val = input[idx];
        output[idx] = __expf(val - block_max) / block_sum;
    }
}

extern "C" void custom_softmax_launcher(float* input, float* output,
                                        int dim,
                                        const int* shape,
                                        const int* strides,
                                        int rank,
                                        cudaStream_t stream) {
    // rank must be 4
    if (rank != 4 || dim < 0 || dim > 3) return;

    int outer_size = 1, inner_size = 1, dim_size = shape[dim];
    for (int i = 0; i < dim; ++i) outer_size *= shape[i];
    for (int i = dim + 1; i < rank; ++i) inner_size *= shape[i];

    unsigned int input_dim_stride = strides[dim];

    dim3 block(256);
    dim3 grid(outer_size, inner_size);

    softmax_dim_kernel<<<grid, block, 0, stream>>>(
        input, output, dim, outer_size, inner_size, dim_size,
        input_dim_stride, input_dim_stride);
}

// ============================================================================
// custom_pixel_shuffle: rearranges input [N, C_in*r^2, H, W] -> output [N, C_in, H*r, W*r]
// Since binding ensures input and output are contiguous, we can use linear indexing.
// ============================================================================

__global__ void pixel_shuffle_kernel(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int N, int C_in, int H, int W, int r,
                                     int in_N_stride, int in_C_stride, int in_H_stride, int in_W_stride,
                                     int out_N_stride, int out_C_stride, int out_H_stride, int out_W_stride) {
    int total = N * C_in * H * r * W * r;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x) {
        // Decompose idx into n, c_out, h_out, w_out
        int n = idx / (C_in * H * r * W * r);
        int rem = idx % (C_in * H * r * W * r);
        int c_out = rem / (H * r * W * r);
        rem %= (H * r * W * r);
        int h_out = rem / (W * r);
        int w_out = rem % (W * r);

        // Input coordinates
        int h_in = h_out / r;
        int w_in = w_out / r;
        int c_in = c_out * r * r + (h_out % r) * r + (w_out % r);

        // Compute input offset using strides (contiguous: in_N_stride = C_in*r^2*H*W, in_C_stride = H*W, in_H_stride = W, in_W_stride = 1)
        int in_offset = n * in_N_stride + c_in * in_C_stride + h_in * in_H_stride + w_in * in_W_stride;
        // Compute output offset using strides (contiguous: out_N_stride = C_in*H*r*W*r, out_C_stride = H*r*W*r, out_H_stride = W*r, out_W_stride = 1)
        int out_offset = n * out_N_stride + c_out * out_C_stride + h_out * out_H_stride + w_out * out_W_stride;

        output[out_offset] = input[in_offset];
    }
}

extern "C" void custom_pixel_shuffle_launcher(float* input,
                                              float* output,
                                              int N, int C_in, int H, int W, int r,
                                              const int* in_strides,
                                              const int* out_strides,
                                              cudaStream_t stream) {
    int total = N * C_in * H * r * W * r;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid cap; kernel uses grid-stride loop

    pixel_shuffle_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C_in, H, W, r,
        in_strides[0], in_strides[1], in_strides[2], in_strides[3],
        out_strides[0], out_strides[1], out_strides[2], out_strides[3]);
}
