// custom_ops.cu
// All CUDA kernels and launchers for diag_embed and l1_norm_dim1.
// No torch dependencies, only raw pointers, scalars, cudaStream_t.

#include <cuda_runtime.h>
#include <cstdint>

// Kernel to embed a [B, N] tensor into a [B, N, N] diagonal tensor.
// output[b, i, j] = input[b, i] if i == j else 0.
__global__ void diag_embed_kernel(float* __restrict__ output,
                                  const float* __restrict__ input,
                                  int B, int N) {
    int total = B * N * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int linear = idx; linear < total; linear += stride) {
        int j = linear % N;
        int tmp = linear / N;
        int i = tmp % N;
        int b = tmp / N;
        if (i == j) {
            output[linear] = input[b * N + i];
        } else {
            output[linear] = 0.0f;
        }
    }
}

// Kernel to compute L1 norm along dim=1 for [B, N, N] tensor.
// output[b, j] = sum_i |input[b, i, j]|.
__global__ void l1_norm_dim1_kernel(float* __restrict__ output,
                                    const float* __restrict__ input,
                                    int B, int N) {
    int total = B * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int linear = idx; linear < total; linear += stride) {
        int j = linear % N;
        int b = linear / N;
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            float val = input[b * N * N + i * N + j];
            sum += fabsf(val);
        }
        output[linear] = sum;
    }
}

// Fused kernel: directly compute output[b, j] = |input[b, j]|,
// because the intermediate tensor is diagonal, and L1 norm along dim=1
// picks only element where i == j.
__global__ void fused_diag_embed_l1_kernel(float* __restrict__ output,
                                           const float* __restrict__ input,
                                           int B, int N) {
    int total = B * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int linear = idx; linear < total; linear += stride) {
        output[linear] = fabsf(input[linear]);
    }
}

// Launcher for diag_embed.
extern "C" void diag_embed_launcher(float* output,
                                    const float* input,
                                    int B, int N,
                                    cudaStream_t stream) {
    int total = B * N * N;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535; // grid-stride loop handles larger
    diag_embed_kernel<<<grid, block, 0, stream>>>(output, input, B, N);
}

// Launcher for l1_norm_dim1.
extern "C" void l1_norm_dim1_launcher(float* output,
                                      const float* input,
                                      int B, int N,
                                      cudaStream_t stream) {
    int total = B * N;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535;
    l1_norm_dim1_kernel<<<grid, block, 0, stream>>>(output, input, B, N);
}

// Fused launcher: input [B, N], output [B, N] = abs(input)
extern "C" void fused_diag_embed_l1_launcher(float* output,
                                             const float* input,
                                             int B, int N,
                                             cudaStream_t stream) {
    int total = B * N;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid > 65535) grid = 65535;
    fused_diag_embed_l1_kernel<<<grid, block, 0, stream>>>(output, input, B, N);
}
