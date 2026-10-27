#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// -----------------------------------------------------------------------------
// Error checking helper for CUDA runtime calls
// -----------------------------------------------------------------------------
static inline void checkCuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        abort();
    }
}
#define CUDA_CHECK(x) checkCuda((x), __FILE__, __LINE__)

// -----------------------------------------------------------------------------
// Kernel: create_linspace
// Generates steps evenly spaced values from start to end inclusive.
// -----------------------------------------------------------------------------
__global__ void create_linspace_kernel(float* output, float start, float end, int64_t steps) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    if (steps <= 1) {
        if (idx == 0) output[0] = start;
        return;
    }
    float denom = (float)(steps - 1);
    for (int64_t i = idx; i < steps; i += stride) {
        float t = (float)i / denom;
        output[i] = fmaf((end - start), t, start);
    }
}

extern "C" void create_linspace_launcher(
    float* output,
    float start,
    float end,
    int64_t steps,
    cudaStream_t stream)
{
    if (steps <= 0) return;
    int blockSize = 256;
    int64_t gridSize64 = (steps + blockSize - 1) / blockSize;
    int gridSize = (gridSize64 > 2147483647LL) ? 2147483647 : (int)gridSize64;
    create_linspace_kernel<<<gridSize, blockSize, 0, stream>>>(
        output, start, end, steps);
    CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// Kernel: tensor_sub
// Computes x[i] - y[j] where x shape [N, C, L] and y shape [L], broadcasting
// y along the last dimension.
// -----------------------------------------------------------------------------
__global__ void tensor_sub_kernel(float* x_in, const float* y, int64_t x_numel, int64_t L) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (int64_t i = idx; i < x_numel; i += stride) {
        int64_t j = i % L;
        x_in[i] -= y[j];
    }
}

extern "C" void tensor_sub_launcher(
    float* x_inout,
    const float* y,
    int64_t x_numel,
    int64_t L,
    cudaStream_t stream)
{
    if (x_numel <= 0 || L <= 0) return;
    int blockSize = 256;
    int64_t gridSize64 = (x_numel + blockSize - 1) / blockSize;
    int gridSize = (gridSize64 > 2147483647LL) ? 2147483647 : (int)gridSize64;
    tensor_sub_kernel<<<gridSize, blockSize, 0, stream>>>(
        x_inout, y, x_numel, L);
    CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// Kernel: conv_transpose1d (pure CUDA implementation)
// input:   [N, C, L]
// weight:  [C, K, R] (PyTorch layout)
// bias:    [K]
// output:  [N, K, L_out]
// L_out = (L-1)*stride - 2*padding + R + output_padding
// -----------------------------------------------------------------------------
__global__ void conv_transpose1d_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t N,
    int C,
    int K,
    int64_t L,
    int R,
    int stride,
    int padding,
    int64_t L_out)
{
    int64_t total = N * K * L_out;
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t grid_stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t pos = idx; pos < total; pos += grid_stride) {
        // decompose pos into (n, k, l_out)
        int64_t l_out = pos % L_out;
        int64_t tmp = pos / L_out;
        int k = tmp % K;
        int n = tmp / K;

        float acc = 0.0f;
        // For each input position l_in and kernel offset r that satisfy:
        //   l_out + padding = l_in * stride + r
        // equivalently: l_in = (l_out + padding - r) / stride if divisible and within [0, L)
        for (int r = 0; r < R; ++r) {
            int64_t numerator = l_out + padding - r;
            if (numerator < 0) continue;
            if (numerator % stride != 0) continue;
            int64_t l_in = numerator / stride;
            if (l_in >= L) continue;
            // accumulate over input channels
            for (int c = 0; c < C; ++c) {
                float x_val = input[((n * C + c) * L) + l_in];
                float w_val = weight[((c * K + k) * R) + r];
                acc += x_val * w_val;
            }
        }
        if (bias != nullptr) {
            acc += bias[k];
        }
        output[pos] = acc;
    }
}

extern "C" void conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t N,
    int C,
    int K,
    int64_t L,
    int R,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream)
{
    if (N <= 0 || C <= 0 || K <= 0 || L <= 0 || R <= 0) return;
    if (stride < 1 || padding < 0 || output_padding < 0) return;

    int64_t L_out = (L - 1) * stride - 2 * padding + R + output_padding;
    if (L_out <= 0) return;

    int blockSize = 256;
    int64_t total = N * K * L_out;
    int64_t gridSize64 = (total + blockSize - 1) / blockSize;
    int gridSize = (gridSize64 > 2147483647LL) ? 2147483647 : (int)gridSize64;

    conv_transpose1d_kernel<<<gridSize, blockSize, 0, stream>>>(
        input, weight, bias, output,
        N, C, K, L, R, stride, padding, L_out);
    CUDA_CHECK(cudaGetLastError());
}
