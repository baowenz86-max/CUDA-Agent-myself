#include <cuda_runtime.h>

// Kernel for L2 normalization along dim=1.
// Input: float tensor of shape [N, C, L]
// Output: same shape, normalized along C (dim=1) for each (n, l).
// p is fixed to 2.0; eps small constant from PyTorch (1e-12) used in F.normalize.
__global__ void normalize_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int L) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * L;
    if (idx >= total) return;

    int n = idx / L;
    int l = idx % L;

    const float* in_ptr = input + (size_t)n * C * L + l;
    float sum_sq = 0.0f;

    // Compute sum of squares over channels
    for (int c = 0; c < C; ++c) {
        float val = in_ptr[(size_t)c * L];
        sum_sq += val * val;
    }

    float norm = sqrtf(sum_sq);
    float denom = fmaxf(norm, 1e-12f);
    float inv_denom = 1.0f / denom;

    float* out_ptr = output + (size_t)n * C * L + l;
    for (int c = 0; c < C; ++c) {
        out_ptr[(size_t)c * L] = in_ptr[(size_t)c * L] * inv_denom;
    }
}

// Launcher for normalize op.
// Expects contiguous input and output tensors of shape [N, C, L] float32.
extern "C" void normalize_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int L,
    cudaStream_t stream) {
    int total = N * L;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    normalize_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, L);
}

// ConvTranspose1d kernel: output[n, co, lo] = bias[co] + sum_{ci, k} input[n, ci, li] * weight[ci, co, k]
// where li = (lo + padding - output_padding - k) / stride, valid if integer and within bounds.
// We use a straightforward mapping: each thread computes one output element.
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias, // may be null
    float* __restrict__ output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (idx >= total) return;

    int lo = idx % L_out;
    int co = (idx / L_out) % C_out;
    int n = idx / (L_out * C_out);

    float acc = 0.0f;
    for (int ci = 0; ci < C_in; ++ci) {
        for (int k = 0; k < kernel_size; ++k) {
            // Compute corresponding input position: li = (lo + padding - k) / stride
            int numerator = lo + padding - k;
            if (numerator < 0) continue;
            if (numerator % stride != 0) continue;
            int li = numerator / stride;
            if (li < 0 || li >= L_in) continue;
            acc += input[((size_t)n * C_in + ci) * L_in + li] *
                   weight[((size_t)ci * C_out + co) * kernel_size + k];
        }
    }
    if (bias != nullptr) {
        acc += bias[co];
    }
    output[idx] = acc;
}

// Conv1d kernel: output[n, co, lo] = bias[co] + sum_{ci_group, k} input[n, ci, li] * weight[co, ci_group, k]
// where ci = (co / (C_out/groups)) * (C_in/groups) + ci_group, li = lo * stride - padding + k.
__global__ void conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias, // may be null
    float* __restrict__ output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int groups) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (idx >= total) return;

    int lo = idx % L_out;
    int co = (idx / L_out) % C_out;
    int n = idx / (L_out * C_out);

    int channels_per_group = C_in / groups;
    int group = co / (C_out / groups);
    int ci_start = group * channels_per_group;

    float acc = 0.0f;
    for (int ci_group = 0; ci_group < channels_per_group; ++ci_group) {
        int ci = ci_start + ci_group;
        for (int k = 0; k < kernel_size; ++k) {
            int li = lo * stride - padding + k;
            if (li >= 0 && li < L_in) {
                acc += input[((size_t)n * C_in + ci) * L_in + li] *
                       weight[((size_t)co * channels_per_group + ci_group) * kernel_size + k];
            }
        }
    }
    if (bias != nullptr) {
        acc += bias[co];
    }
    output[idx] = acc;
}

// Launcher for conv_transpose1d
// Assumes input, weight contiguous, bias optional and contiguous if provided.
extern "C" void conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias, // can be nullptr
    float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int output_padding,
    cudaStream_t stream) {
    int total = N * C_out * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output, N, C_in, C_out, L_in, L_out,
        kernel_size, stride, padding, output_padding);
}

// Launcher for conv1d
extern "C" void conv1d_launcher(
    const float* input,
    const float* weight,
    const float* bias, // can be nullptr
    float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int kernel_size, int stride, int padding, int groups,
    cudaStream_t stream) {
    int total = N * C_out * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output, N, C_in, C_out, L_in, L_out,
        kernel_size, stride, padding, groups);
}
