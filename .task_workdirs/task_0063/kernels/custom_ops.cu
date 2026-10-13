#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

// Adaptive average pooling 1D kernel
// Each output element is the mean of pooled input elements in the adaptive window.
// The input tensor is assumed to be contiguous with shape [N, C, L_in].
// Output has shape [N, C, L_out].
// Grid: one block per (n, c) pair, threads across L_out.
__global__ void adaptive_avgpool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int L_in,
    int L_out)
{
    int n = blockIdx.x;
    int c = blockIdx.y;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Base index for this (n,c) slice
    const float* in_slice = input + (size_t)n * gridDim.y * L_in; // gridDim.y == C
    float* out_slice = output + (size_t)n * gridDim.y * L_out;

    for (int i = tid; i < L_out; i += stride) {
        int start = (int)floorf((float)i * L_in / L_out);
        int end = (int)ceilf((float)(i + 1) * L_in / L_out);
        if (end > L_in) end = L_in;
        float sum = 0.0f;
        for (int j = start; j < end; ++j) {
            sum += in_slice[c * L_in + j];
        }
        out_slice[c * L_out + i] = sum / (end - start);
    }
}

// Launcher for adaptive_avgpool1d_forward
// Input: x is a contiguous float32 CUDA tensor of shape [N, C, L_in]
// output_size: desired L_out
// Returns: newly allocated float32 tensor of shape [N, C, L_out] via external allocation.
// The caller (binding layer) is responsible for allocating output and passing pointers.
extern "C" void adaptive_avgpool1d_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int L_in,
    int L_out,
    cudaStream_t stream)
{
    // Thread configuration: one block per (n,c) pair, with enough threads for striding over L_out.
    dim3 grid(N, C);
    int threads = 256;
    if (L_out < threads) threads = L_out; // ensure at least one thread, but if L_out=0 no launch needed
    if (N <= 0 || C <= 0 || L_in <= 0 || L_out <= 0) return;
    adaptive_avgpool1d_kernel<<<grid, threads, 0, stream>>>(input, output, L_in, L_out);
}

// Unfold (im2col) kernel
// Input: x shape [N, C, L], contiguous float32
// Parameters: kernel_size, stride, padding (all int)
// Output shape: [N, C * kernel_size, L_out]
// L_out = floor((L + 2*padding - kernel_size) / stride) + 1
// Output layout: for each n, output rows are c * kernel_size + k,
// and columns are output positions.
// Each thread computes one output element.
__global__ void unfold_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int L,
    int kernel_size,
    int stride,
    int padding,
    int L_out)
{
    // Total output elements: N * C * kernel_size * L_out
    int total = N * C * kernel_size * L_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Decompose idx into (n, ck, l_out)
    int l_out = idx % L_out;
    int tmp = idx / L_out;
    int ck = tmp % (C * kernel_size);
    int n = tmp / (C * kernel_size);

    // Decompose ck into c and k
    int c = ck / kernel_size;
    int k = ck % kernel_size;

    // Input index: n*C*L + c*L + (l_out*stride - padding + k)
    int in_idx = l_out * stride - padding + k;
    if (in_idx >= 0 && in_idx < L) {
        output[idx] = input[(size_t)n * C * L + c * L + in_idx];
    } else {
        output[idx] = 0.0f;
    }
}

// Launcher for unfold_forward
// Input: x contiguous float32 [N, C, L]
// kernel_size, stride, padding
// Output: float32 [N, C*kernel_size, L_out]
extern "C" void unfold_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int L,
    int kernel_size,
    int stride,
    int padding,
    int L_out,
    cudaStream_t stream)
{
    if (N <= 0 || C <= 0 || L <= 0 || kernel_size <= 0 || stride <= 0 || L_out <= 0) return;
    int total = N * C * kernel_size * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    unfold_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, L, kernel_size, stride, padding, L_out);
}
