#include <cuda_runtime.h>

// ======================= Conv1d =======================
__global__ void conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int L,
    int C_out, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C_out * (L - K + 1);
    int stride = blockDim.x * gridDim.x;
    for (int pos = idx; pos < total; pos += stride) {
        int b = pos / (C_out * (L - K + 1));
        int rem = pos % (C_out * (L - K + 1));
        int oc = rem / (L - K + 1);
        int ol = rem % (L - K + 1);
        float acc = bias[oc];
        for (int k = 0; k < K; ++k) {
            int il = ol + k;
            for (int ic = 0; ic < C_in; ++ic) {
                acc += input[(b * C_in + ic) * L + il] *
                       weight[(oc * C_in + ic) * K + k];
            }
        }
        output[pos] = acc;
    }
}

extern "C" void conv1d_launcher(
    const float* input, const float* weight, const float* bias,
    float* output, int B, int C_in, int L, int C_out, int K,
    cudaStream_t stream) {
    int L_out = L - K + 1;
    int total = B * C_out * L_out;
    int block = 256;
    int grid = (total + block - 1) / block;
    conv1d_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output, B, C_in, L, C_out, K);
}

// ======================= Batch MatMul (A @ A^T) =======================
__global__ void batch_matmul_kernel(
    const float* __restrict__ A,
    float* __restrict__ out,
    int B, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * M * M;
    int stride = blockDim.x * gridDim.x;
    for (int pos = idx; pos < total; pos += stride) {
        int b = pos / (M * M);
        int rem = pos % (M * M);
        int i = rem / M;
        int j = rem % M;
        float acc = 0.0f;
        for (int k = 0; k < N; ++k) {
            acc += A[(b * M + i) * N + k] * A[(b * M + j) * N + k];
        }
        out[pos] = acc;
    }
}

extern "C" void batch_matmul_launcher(
    const float* A, float* out,
    int B, int M, int N,
    cudaStream_t stream) {
    int total = B * M * M;
    int block = 256;
    int grid = (total + block - 1) / block;
    batch_matmul_kernel<<<grid, block, 0, stream>>>(A, out, B, M, N);
}

// ======================= Softplus =======================
__global__ void softplus_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < count; i += stride) {
        float x = input[i];
        // PyTorch: softplus(x) = log(1 + exp(x)) with threshold 20
        if (x > 20.0f) {
            output[i] = x;
        } else {
            output[i] = log1pf(expf(x));
        }
    }
}

extern "C" void softplus_launcher(
    const float* input, float* output, int count,
    cudaStream_t stream) {
    int block = 256;
    int grid = (count + block - 1) / block;
    softplus_kernel<<<grid, block, 0, stream>>>(input, output, count);
}

// ======================= Add Bias =======================
__global__ void add_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C * N;
    int stride = blockDim.x * gridDim.x;
    for (int pos = idx; pos < total; pos += stride) {
        int b = pos / (C * N);
        int rem = pos % (C * N);
        int c = rem / N;
        int n = rem % N;
        output[pos] = input[pos] + bias[c];
    }
}

extern "C" void add_bias_launcher(
    const float* input, const float* bias, float* output,
    int B, int C, int N,
    cudaStream_t stream) {
    int total = B * C * N;
    int block = 256;
    int grid = (total + block - 1) / block;
    add_bias_kernel<<<grid, block, 0, stream>>>(input, bias, output, B, C, N);
}
