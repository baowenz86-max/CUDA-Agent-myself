// CUDA kernels for custom ops
// This file contains non-convolutional CUDA kernels.
// Conv/ConvTranspose are handled by cuDNN in the binding layer.

#include <cuda_runtime.h>
#include <math.h>

// Kernel to compute logsumexp along the last dimension.
// input: [B, C, L] (or any shape with last dim L)
// output: [B, C, 1] if keepdim=true, else [B, C]
// Each thread handles one (b, c) pair and reduces over L.
__global__ void logsumexp_last_dim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int L,
    int input_stride_b, int input_stride_c,
    bool keepdim,
    int output_stride_b, int output_stride_c
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C;
    if (idx >= total) return;

    int b = idx / C;
    int c = idx % C;

    const float* data = input + b * input_stride_b + c * input_stride_c;
    float max_val = -1e30f; // -inf
    for (int i = 0; i < L; ++i) {
        float v = data[i];
        if (v > max_val) max_val = v;
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < L; ++i) {
        sum_exp += __expf(data[i] - max_val);
    }

    float result = max_val + logf(sum_exp);

    if (keepdim) {
        output[b * output_stride_b + c * output_stride_c] = result;
    } else {
        output[b * C + c] = result;
    }
}

// Launcher for logsumexp along the last dimension (dim = input.dim()-1)
extern "C" void logsumexp_dim_forward_launcher(
    const float* input,
    float* output,
    int B,           // batch size
    int C,           // channels
    int L,           // length
    int input_stride_b,
    int input_stride_c,
    bool keepdim,
    int output_stride_b,
    int output_stride_c,
    cudaStream_t stream
) {
    int total = B * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    logsumexp_last_dim_kernel<<<blocks, threads, 0, stream>>>(
        input, output, B, C, L,
        input_stride_b, input_stride_c,
        keepdim,
        output_stride_b, output_stride_c
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error handling is usually done by caller; here we can just ignore or record.
        // No torch dependency, so we cannot throw exceptions. We use a printf for debugging.
        // In production, this launcher should be called with a valid stream and checked later.
    }
}

// Optional: global logsumexp along any dimension could be added, but only last dim needed.
