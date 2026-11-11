#include <cuda_runtime.h>

// CUDA kernel for concatenating two tensors along dim=1.
// Each tensor has shape [N, C, H, W] with C = 8, concatenated C_out = 16.
// Since tensors are contiguous, output is contiguous.
// We assume both inputs are contiguous, otherwise binding layer should make contiguous copies.
// For arbitrary shapes, we can launch a 1D kernel with total elements = N*C_out*H*W.
// The kernel computes output index and copies from appropriate input.
__global__ void cat_kernel(float* __restrict__ output,
                           const float* __restrict__ input1,
                           const float* __restrict__ input2,
                           int N, int C_in, int C_out, int H, int W) {
    int total = N * C_out * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int plane = H * W;
    int in_plane_offset = C_in * plane;
    for (int i = idx; i < total; i += stride) {
        // Compute coordinates
        int w = i % W;
        int h = (i / W) % H;
        int c = (i / (W * H)) % C_out;
        int n = i / (W * H * C_out);
        if (c < C_in) {
            int src_idx = ((n * C_in + c) * H + h) * W + w;
            output[i] = input1[src_idx];
        } else {
            int c2 = c - C_in;
            int src_idx = ((n * C_in + c2) * H + h) * W + w;
            output[i] = input2[src_idx];
        }
    }
}

// CUDA kernel for element-wise multiplication with scalar multiplier.
// multiplier is expected to have at least one element, we read multiplier[0].
__global__ void mul_scalar_kernel(float* __restrict__ output,
                                  const float* __restrict__ input,
                                  const float* __restrict__ multiplier,
                                  int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float scalar = multiplier[0];
    for (int i = idx; i < total; i += stride) {
        output[i] = input[i] * scalar;
    }
}

// Launcher for concatenation: out = cat([in1, in2], dim=1)
// Inputs are pointer to contiguous float arrays, sizes: in1 and in2 have same shape [N, C_in, H, W]
// Output shape [N, C_out, H, W] with C_out = 2*C_in.
// N, C_in, H, W are int.
extern "C" void custom_cat_launcher(float* output,
                                   const float* input1,
                                   const float* input2,
                                   int N, int C_in, int H, int W,
                                   cudaStream_t stream) {
    int C_out = 2 * C_in;
    int total = N * C_out * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid limit
    cat_kernel<<<blocks, threads, 0, stream>>>(output, input1, input2, N, C_in, C_out, H, W);
}

// Launcher for multiplication: out = x * multiplier (broadcast scalar)
// x shape: [N, C, H, W] total = N*C*H*W
extern "C" void custom_mul_launcher(float* output,
                                    const float* input,
                                    const float* multiplier,
                                    int total,
                                    cudaStream_t stream) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    mul_scalar_kernel<<<blocks, threads, 0, stream>>>(output, input, multiplier, total);
}
