// custom_ops.cu - CUDA kernels for custom_unfold_forward
#include <cuda_runtime.h>
#include <cstdint>

// Unfold kernel: output[n, c*K + ki, i*outW + j] = input[n, c, i*strideH + ki*dilationH, j*strideW + kj*dilationW]
// K = kernelH * kernelW
template <typename T>
__global__ void unfold_forward_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t kernelH,
    int64_t kernelW,
    int64_t dilationH,
    int64_t dilationW,
    int64_t strideH,
    int64_t strideW,
    int64_t outH,
    int64_t outW,
    int64_t L,
    int64_t K)
{
    int64_t total = N * C * K * L;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t linear = idx; linear < total; linear += stride) {
        // Decompose linear index
        int64_t tmp = linear;
        int64_t j = tmp % outW;
        tmp /= outW;
        int64_t i = tmp % outH;
        tmp /= outH;
        int64_t k = tmp % K;
        tmp /= K;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        int64_t kh = k / kernelW;
        int64_t kw = k % kernelW;

        int64_t in_h = i * strideH + kh * dilationH;
        int64_t in_w = j * strideW + kw * dilationW;

        // No padding, so input indices are valid if H/W large enough (verified by binding)
        output[linear] = input[((n * C + c) * H + in_h) * W + in_w];
    }
}

// C-style launcher for unfold
extern "C" void custom_unfold_forward_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t kernelH,
    int64_t kernelW,
    int64_t dilationH,
    int64_t dilationW,
    int64_t strideH,
    int64_t strideW,
    int64_t outH,
    int64_t outW,
    int64_t L,
    int64_t K,
    cudaStream_t stream)
{
    int64_t total = N * C * K * L;
    int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    if (blocks > 1073741824LL) blocks = 1073741824LL; // cap grid size
    unfold_forward_kernel<float><<<(unsigned)blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, kernelH, kernelW,
        dilationH, dilationW, strideH, strideW, outH, outW, L, K);
}

// Placeholder for custom_reshape_3d: no kernel needed, binding will use torch::reshape
// This declaration ensures the symbol exists for potential linking.
extern "C" void custom_reshape_3d_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream)
{
    // No-op: reshape is metadata-only.
}
