// kernels/custom_ops.cu
// CUDA kernels for custom_tril, custom_elu, custom_add_bias
// All launchers take raw pointers and cudaStream_t; no torch/ATen/c10 dependencies.

#include <cuda_runtime.h>
#include <math.h>

// ------------------------- custom_tril -------------------------
// Applies torch.tril to a 4D tensor with shape [B, C, H, W].
// Zeroes elements where j > i in the last two dimensions.
// Input is assumed contiguous; flattened index = ((b*C + c)*H + i)*W + j
__global__ void tril_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int B, int C, int H, int W,
                            long long total_elements) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += stride) {
        // Decompose flattened index
        int j = idx % W;
        long long tmp = idx / W;
        int i = tmp % H;
        tmp /= H;
        // c = tmp % C; b = tmp / C (unused)
        // We only need i and j
        if (j <= i) {
            output[idx] = input[idx];
        } else {
            output[idx] = 0.0f;
        }
    }
}

extern "C" void custom_tril_launcher(const float* input,
                                     float* output,
                                     int B, int C, int H, int W,
                                     long long total_elements,
                                     cudaStream_t stream) {
    const int threads = 256;
    long long blocks_ll = (total_elements + threads - 1) / threads;
    // Cap blocks to max grid dimension (2^31-1) and use grid-stride loop
    int blocks = (blocks_ll > 2147483647LL) ? 2147483647 : (int)blocks_ll;
    tril_kernel<<<blocks, threads, 0, stream>>>(input, output, B, C, H, W, total_elements);
}

// ------------------------- custom_elu -------------------------
// Applies ELU with alpha=1.0 elementwise: output = x if x > 0 else exp(x)-1
__global__ void elu_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           long long total_elements) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += stride) {
        float val = input[idx];
        output[idx] = (val > 0.0f) ? val : expf(val) - 1.0f;
    }
}

extern "C" void custom_elu_launcher(const float* input,
                                    float* output,
                                    long long total_elements,
                                    cudaStream_t stream) {
    const int threads = 256;
    long long blocks_ll = (total_elements + threads - 1) / threads;
    int blocks = (blocks_ll > 2147483647LL) ? 2147483647 : (int)blocks_ll;
    elu_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements);
}

// ------------------------- custom_add_bias -------------------------
// Adds bias[c] to each element in channel c. bias has shape [C,1,1].
// input shape [B, C, H, W]; output same shape.
// For each element (b,c,i,j): output = input + bias[c]
__global__ void add_bias_kernel(const float* __restrict__ input,
                                const float* __restrict__ bias,
                                float* __restrict__ output,
                                int B, int C, int H, int W,
                                long long total_elements) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += stride) {
        // Decompose to get channel c
        int j = idx % W;
        long long tmp = idx / W;
        int i = tmp % H;
        tmp /= H;
        int c = tmp % C;
        // b = tmp / C;
        float b = bias[c];  // bias is contiguous of length C
        output[idx] = input[idx] + b;
    }
}

extern "C" void custom_add_bias_launcher(const float* input,
                                         const float* bias,
                                         float* output,
                                         int B, int C, int H, int W,
                                         long long total_elements,
                                         cudaStream_t stream) {
    const int threads = 256;
    long long blocks_ll = (total_elements + threads - 1) / threads;
    int blocks = (blocks_ll > 2147483647LL) ? 2147483647 : (int)blocks_ll;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(input, bias, output, B, C, H, W, total_elements);
}
