#include <cuda_runtime.h>
#include <cfloat>

// Softmax over dim=1: input shape [B, C, L], contiguous or strided
// Each thread block processes one (b, l) pair, computing softmax over C channels.
// grid = B * L, block = min(C, 256) threads.
// Supports arbitrary strides as long as data_ptr is CUDA pointer and stride values are int64_t.
__global__ void softmax_dim1_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int B, int C, int L,
                                    int64_t in_stride_b, int64_t in_stride_c, int64_t in_stride_l,
                                    int64_t out_stride_b, int64_t out_stride_c, int64_t out_stride_l) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * L) return;

    int b = idx / L;
    int l = idx - b * L;

    const float* in_base = input + b * in_stride_b + l * in_stride_l;
    float* out_base = output + b * out_stride_b + l * out_stride_l;

    // Find max for numerical stability
    float max_val = -FLT_MAX;
    for (int c = 0; c < C; ++c) {
        float v = in_base[c * in_stride_c];
        if (v > max_val) max_val = v;
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        float v = __expf(in_base[c * in_stride_c] - max_val);
        out_base[c * out_stride_c] = v;
        sum += v;
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (int c = 0; c < C; ++c) {
        out_base[c * out_stride_c] *= inv_sum;
    }
}

// Launcher for custom_softmax_dim1
// input and output must be CUDA float32 pointers, can be non-contiguous (strides provided)
// B, C, L are positive integers; no CPU pointers allowed.
extern "C" void custom_softmax_dim1_launcher(const float* input,
                                              float* output,
                                              int B, int C, int L,
                                              int64_t in_stride_b, int64_t in_stride_c, int64_t in_stride_l,
                                              int64_t out_stride_b, int64_t out_stride_c, int64_t out_stride_l,
                                              cudaStream_t stream) {
    int total_pairs = B * L;
    int threads = (C < 256) ? C : 256;
    int blocks = (total_pairs + threads - 1) / threads;
    if (blocks == 0 || threads == 0) return;

    softmax_dim1_kernel<<<blocks, threads, 0, stream>>>(
        input, output, B, C, L,
        in_stride_b, in_stride_c, in_stride_l,
        out_stride_b, out_stride_c, out_stride_l);
}

// Declaration for custom_conv_transpose1d launcher (implemented in binding.cpp using cuDNN).
// The binding will call this function with raw pointers and cuDNN handles/descriptors.
// This .cu file does not implement convolution; only declares the interface to satisfy linker.
extern "C" void custom_conv_transpose1d_launcher(const float* input,
                                                  const float* weight,
                                                  float* output,
                                                  int batch,
                                                  int in_channels,
                                                  int out_channels,
                                                  int in_length,
                                                  int out_length,
                                                  int kernel_size,
                                                  int stride,
                                                  int padding,
                                                  int output_padding,
                                                  cudaStream_t stream);
