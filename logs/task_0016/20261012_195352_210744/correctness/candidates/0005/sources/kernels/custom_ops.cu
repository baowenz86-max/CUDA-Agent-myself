#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// addmm: C = A @ B^T + bias, with A (M x K) row-major, B (N x K) row-major,
// and flexible leading dimensions to handle non-contiguous views (e.g., weight.t()).
// -----------------------------------------------------------------------------
__global__ void addmm_kernel(
    const float* __restrict__ bias,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    int ldA, int ldB) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = bias[col];
        for (int k = 0; k < K; ++k) {
            sum += A[row * ldA + k] * B[col * ldB + k];
        }
        C[row * N + col] = sum;
    }
}

extern "C" void addmm_forward_launcher(
    const float* bias,
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    int ldA, int ldB,
    cudaStream_t stream) {
    dim3 block(16, 1);
    dim3 grid((N + block.x - 1) / block.x, M);
    addmm_kernel<<<grid, block, 0, stream>>>(
        bias, A, B, C, M, N, K, ldA, ldB);
}

// -----------------------------------------------------------------------------
// unsqueeze: simply copies input to output with shape [1, M, N].
// Since the tensor has the same number of elements, we use a 1D copy.
// -----------------------------------------------------------------------------
__global__ void unsqueeze_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = input[idx];
    }
}

extern "C" void unsqueeze_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    unsqueeze_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

// -----------------------------------------------------------------------------
// interpolate_linear: 3D input [1, M, N_in] -> output [1, M, N_out],
// with scale_factor (float), align_corners=false.
// For each output column j, compute src = (j + 0.5) / scale_factor - 0.5,
// clamp to [0, N_in-1], and linearly interpolate between floor and ceil.
// -----------------------------------------------------------------------------
__device__ __forceinline__ float linear_interpolate(
    float src, int N_in, float scale) {
    if (src < 0.f) src = 0.f;
    float max_val = (float)(N_in - 1);
    if (src > max_val) src = max_val;
    int left = (int)floorf(src);
    int right = min(left + 1, N_in - 1);
    float frac = src - left;
    return (1.f - frac) * (left) + frac * (right); // Placeholder, actual kernel uses input
}

__global__ void interpolate_linear_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M, int N_in, int N_out,
    float scale_factor) {
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N_out) {
        float src = ((float)col + 0.5f) / scale_factor - 0.5f;
        if (src < 0.f) src = 0.f;
        float max_val = (float)(N_in - 1);
        if (src > max_val) src = max_val;
        int left = (int)floorf(src);
        int right = left + 1;
        if (right >= N_in) right = N_in - 1;
        float frac = src - left;
        const float* row_ptr = input + row * N_in;
        float val = (1.f - frac) * row_ptr[left] + frac * row_ptr[right];
        output[row * N_out + col] = val;
    }
}

extern "C" void interpolate_linear_forward_launcher(
    const float* input,
    float* output,
    int M, int N_in, int N_out,
    float scale_factor,
    cudaStream_t stream) {
    dim3 block(256, 1);
    dim3 grid((N_out + block.x - 1) / block.x, M);
    interpolate_linear_kernel<<<grid, block, 0, stream>>>(
        input, output, M, N_in, N_out, scale_factor);
}
