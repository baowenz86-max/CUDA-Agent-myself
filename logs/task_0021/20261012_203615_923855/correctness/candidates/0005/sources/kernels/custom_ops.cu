// custom_ops.cu : all CUDA kernels and launchers for model_new.py
// No torch/ATen/c10/pybind dependencies; only raw pointers, scalars, cudaStream_t.
#include <cuda_runtime.h>
#include <math.h>

// -------------------------- Kernel definitions --------------------------

// Conv1d: in_channels=3, out_channels=64, kernel_size=3, stride=1, no padding, dilation=1.
// Input: [B, C_in, L_in], weight: [C_out, C_in, K], bias: [C_out] (may be null)
// Output: [B, C_out, L_out] where L_out = L_in - K + 1 = L_in - 2.
// Each thread computes one output element (b, co, l).
__global__ void conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int L_in, int C_out, int K, int L_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C_out * L_out;
    if (idx >= total) return;
    
    int l = idx % L_out;
    int co = (idx / L_out) % C_out;
    int b = idx / (L_out * C_out);
    
    float acc = (bias != nullptr) ? bias[co] : 0.0f;
    for (int ci = 0; ci < C_in; ++ci) {
        for (int k = 0; k < K; ++k) {
            acc += input[b * C_in * L_in + ci * L_in + (l + k)] *
                   weight[co * C_in * K + ci * K + k];
        }
    }
    output[idx] = acc;
}

// Batch matmul A @ A^T: input [B, C, L], output [B, C, C]
// Each thread computes one output element (b, i, j).
// Use shared memory tile of A rows to improve performance: each block handles a tile of C length? Since C=64 and L=8190, we can use a 1D grid with each thread computing multiple output elements.
// Simple kernel: each thread computes one output element with a loop over L.
__global__ void batch_matmul_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int L
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * C;
    if (idx >= total) return;
    
    int j = idx % C;
    int i = (idx / C) % C;
    int b = idx / (C * C);
    
    const float* A_i = input + b * C * L + i * L;
    const float* A_j = input + b * C * L + j * L; // A^T columns are A rows, but multiplication sum over L
    float sum = 0.0f;
    for (int k = 0; k < L; ++k) {
        sum += A_i[k] * A_j[k];
    }
    output[idx] = sum;
}

// Softplus: y = log(1 + exp(x)), with linear for x > 20 to avoid overflow.
__global__ void softplus_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = input[idx];
    output[idx] = (x > 20.0f) ? x : log1p(expf(x));
}

// Add bias: bias shape [C] (or [C,1] but contiguous), broadcast over last dimension of input [B, C, N]
__global__ void add_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * N;
    if (idx >= total) return;
    int n = idx % N;
    int c = (idx / N) % C;
    output[idx] = input[idx] + bias[c];
}

// -------------------------- Host launchers (extern "C") --------------------------

extern "C" void conv1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int C_in, int L_in, int C_out, int K, int L_out,
    cudaStream_t stream
) {
    int total = B * C_out * L_out;
    int block = 256;
    int grid = (total + block - 1) / block;
    conv1d_kernel<<<grid, block, 0, stream>>>(input, weight, bias, output,
                                               B, C_in, L_in, C_out, K, L_out);
}

extern "C" void batch_matmul_launcher(
    const float* input,
    float* output,
    int B, int C, int L,
    cudaStream_t stream
) {
    int total = B * C * C;
    int block = 256;
    int grid = (total + block - 1) / block;
    batch_matmul_kernel<<<grid, block, 0, stream>>>(input, output, B, C, L);
}

extern "C" void softplus_launcher(
    const float* input,
    float* output,
    int n,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (n + block - 1) / block;
    softplus_kernel<<<grid, block, 0, stream>>>(input, output, n);
}

extern "C" void add_bias_launcher(
    const float* input,
    const float* bias,
    float* output,
    int B, int C, int N,
    cudaStream_t stream
) {
    int total = B * C * N;
    int block = 256;
    int grid = (total + block - 1) / block;
    add_bias_kernel<<<grid, block, 0, stream>>>(input, bias, output, B, C, N);
}
