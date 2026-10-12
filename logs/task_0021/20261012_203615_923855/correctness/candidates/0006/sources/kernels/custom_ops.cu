#include <cuda_runtime.h>
#include <math.h>

// All kernels are float32 only as per contract.

// -----------------------------------------------------------------------------
// Conv1d forward: output[b, oc, ol] = bias[oc] + sum_{ic, k} input[b, ic, ol+k] * weight[oc, ic, k]
// input is [B, C_in, L_in], weight [C_out, C_in, K], output [B, C_out, L_out]
// Assume contiguous tensors, stride: input: (C_in*L_in, L_in, 1), weight: (C_in*K, K, 1), output: (C_out*L_out, L_out, 1)
// No padding, stride=1, dilation=1, L_out = L_in - K + 1.
// B, C_in, C_out, L_in, L_out are ints passed from binding.
__global__ void conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int C_out, int L_in, int L_out, int K) {
    int total = B * C_out * L_out;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = index; idx < total; idx += stride) {
        int ol = idx % L_out;
        int tmp = idx / L_out;
        int oc = tmp % C_out;
        int b = tmp / C_out;
        float acc = bias[oc];
        for (int ic = 0; ic < C_in; ++ic) {
            const float* in_ptr = input + b * C_in * L_in + ic * L_in + ol;
            const float* w_ptr = weight + oc * C_in * K + ic * K;
            #pragma unroll
            for (int k = 0; k < K; ++k) {
                acc += in_ptr[k] * w_ptr[k];
            }
        }
        output[idx] = acc;
    }
}

extern "C" void conv1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int C_in, int C_out, int L_in, int L_out, int K,
    cudaStream_t stream) {
    int total = B * C_out * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Limit blocks to avoid excessive grid size if total huge; but total ~67M, blocks ~262k ok.
    conv1d_kernel<<<blocks, threads, 0, stream>>>(input, weight, bias, output, B, C_in, C_out, L_in, L_out, K);
}

// -----------------------------------------------------------------------------
// Batch Matmul: output[b, i, j] = sum_k A[b, i, k] * A[b, j, k]
// A is [B, N, K], output [B, N, N]
// Because A^T is used, we multiply rows of A with rows of A (dot product over K).
__global__ void batch_matmul_kernel(
    const float* __restrict__ A,
    float* __restrict__ output,
    int B, int N, int K) {
    int total = B * N * N;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = index; idx < total; idx += stride) {
        int j = idx % N;
        int tmp = idx / N;
        int i = tmp % N;
        int b = tmp / N;
        const float* row_i = A + b * N * K + i * K;
        const float* row_j = A + b * N * K + j * K;
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += row_i[k] * row_j[k];
        }
        output[idx] = acc;
    }
}

extern "C" void batch_matmul_launcher(
    const float* A,
    float* output,
    int B, int N, int K,
    cudaStream_t stream) {
    int total = B * N * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    batch_matmul_kernel<<<blocks, threads, 0, stream>>>(A, output, B, N, K);
}

// -----------------------------------------------------------------------------
// Softplus: output = log(1 + exp(x)), with linear fallback for x > threshold
// PyTorch default: beta=1, threshold=20
__global__ void softplus_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total,
    float threshold) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = index; idx < total; idx += stride) {
        float x = input[idx];
        float y;
        if (x > threshold) {
            y = x;
        } else {
            y = log1pf(expf(x));
        }
        output[idx] = y;
    }
}

extern "C" void softplus_launcher(
    const float* input,
    float* output,
    int total,
    float threshold,
    cudaStream_t stream) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    softplus_kernel<<<blocks, threads, 0, stream>>>(input, output, total, threshold);
}

// -----------------------------------------------------------------------------
// Add bias: output[b, c, n] = input[b, c, n] + bias[c]
// bias shape [C_out, 1] but we treat as array of length C_out; broadcast over last dim.
__global__ void add_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_out, int N) {
    int total = B * C_out * N;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int idx = index; idx < total; idx += stride) {
        int n = idx % N;
        int tmp = idx / N;
        int c = tmp % C_out;
        // b = tmp / C_out not used except for indexing input, but input has same layout with leading B dim.
        output[idx] = input[idx] + bias[c];
    }
}

extern "C" void add_bias_launcher(
    const float* input,
    const float* bias,
    float* output,
    int B, int C_out, int N,
    cudaStream_t stream) {
    int total = B * C_out * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(input, bias, output, B, C_out, N);
}
