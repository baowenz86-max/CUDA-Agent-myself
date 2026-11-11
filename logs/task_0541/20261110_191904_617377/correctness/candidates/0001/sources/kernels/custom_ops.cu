// kernels/custom_ops.cu
// Custom CUDA kernels for the model.
// Provides two elementwise kernels:
// 1. BatchNorm2d eval mode (per-channel affine transform)
// 2. GELU activation (exact erf)
// All kernels are designed for float32 inputs on CUDA.

#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Kernel 1: eval-mode BatchNorm2d
// Computes y = (x - running_mean[c]) * inv_std[c] * weight[c] + bias[c]
// where inv_std[c] = rsqrt(running_var[c] + eps)
// Input is non-contiguous with shape [N, C, H, W] and strides [sN, sC, sH, sW].
// Output is assumed contiguous (same layout as torch::empty_like).
//
// We use a grid-stride loop over all elements. Each element is mapped to
// (n, c, h, w) using the linear index and dimension sizes.
// Thread indexing: 1D flatten over N*C*H*W.
// -----------------------------------------------------------------------------
__global__ void batchnorm2d_eval_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N, int C, int H, int W,
    long long input_sN, long long input_sC, long long input_sH, long long input_sW) {

    long long total = (long long)N * C * H * W;
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; index < total; index += stride) {
        // Decode n, c, h, w from linear index assuming row-major NCHW order
        long long tmp = index;
        int w = (int)(tmp % W);
        tmp /= W;
        int h = (int)(tmp % H);
        tmp /= H;
        int c = (int)(tmp % C);
        int n = (int)(tmp / C);

        long long input_offset = (long long)n * input_sN + (long long)c * input_sC
                                 + (long long)h * input_sH + (long long)w * input_sW;

        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float gamma = weight[c];
        float beta = bias[c];

        float val = input[input_offset];
        float normalized = (val - mean) * inv_std;
        output[index] = normalized * gamma + beta;
    }
}

// -----------------------------------------------------------------------------
// Kernel 2: GELU activation (exact)
// Computes y = x * 0.5 * (1 + erf(x / sqrt(2)))
// Non-contiguous input with strides; output contiguous.
// -----------------------------------------------------------------------------
__global__ void gelu_forward_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long numel,
    long long input_s0, long long input_s1, long long input_s2, long long input_s3,
    int ndims,
    int dim0, int dim1, int dim2, int dim3) {

    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    // Constant sqrt(0.5) multiplied into the argument or use precomputed sqrt(2) inverse.
    // We use 0.70710678118654752440f = 1/sqrt(2) to reduce operations.
    const float inv_sqrt2 = 0.70710678118654752440f;

    for (; index < numel; index += stride) {
        // Decode linear index to n,c,h,w. ndims is expected to be 4 always here.
        // We support only 4D inputs for simplicity; the contract guarantees 4D.
        long long tmp = index;
        int w = (int)(tmp % dim3);
        tmp /= dim3;
        int h = (int)(tmp % dim2);
        tmp /= dim2;
        int c = (int)(tmp % dim1);
        int n = (int)(tmp / dim1);

        long long input_offset = (long long)n * input_s0 + (long long)c * input_s1
                                 + (long long)h * input_s2 + (long long)w * input_s3;

        float x = input[input_offset];
        float y = x * 0.5f * (1.0f + erff(x * inv_sqrt2));
        output[index] = y;
    }
}

// -----------------------------------------------------------------------------
// Host launchers (extern "C", no PyTorch/ATen types)
// -----------------------------------------------------------------------------

// Launcher for BatchNorm2d eval.
// Parameters:
//   output: pointer to output tensor (contiguous float32)
//   input: pointer to input tensor (float32, may be non-contiguous)
//   weight, bias, running_mean, running_var: contiguous float32 arrays of length C
//   eps: float epsilon
//   N, C, H, W: dimensions
//   input_sN, input_sC, input_sH, input_sW: strides of input in elements
//   stream: CUDA stream to launch on
// The output is assumed to be contiguous with strides (C*H*W, H*W, W, 1).
extern "C" void custom_batchnorm2d_eval_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t input_sN, int64_t input_sC, int64_t input_sH, int64_t input_sW,
    cudaStream_t stream) {

    long long total = (long long)N * C * H * W;
    if (total <= 0) return;

    int threads = 256;
    long long blocks_ll = (total + threads - 1) / threads;
    int blocks = (int)min(blocks_ll, (long long)2147483647);

    batchnorm2d_eval_kernel<<<blocks, threads, 0, stream>>>(
        output, input, weight, bias, running_mean, running_var,
        (float)eps, (int)N, (int)C, (int)H, (int)W,
        (long long)input_sN, (long long)input_sC, (long long)input_sH, (long long)input_sW);
}

// Launcher for GELU.
// Parameters analogous; output is contiguous, input may be non-contiguous.
// The input dimensions are assumed 4D.
extern "C" void custom_gelu_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    int64_t input_s0, int64_t input_s1, int64_t input_s2, int64_t input_s3,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
    cudaStream_t stream) {

    if (numel <= 0) return;

    int threads = 256;
    long long blocks_ll = ((long long)numel + threads - 1) / threads;
    int blocks = (int)min(blocks_ll, (long long)2147483647);

    gelu_forward_kernel<<<blocks, threads, 0, stream>>>(
        output, input, (long long)numel,
        (long long)input_s0, (long long)input_s1, (long long)input_s2, (long long)input_s3,
        4, (int)dim0, (int)dim1, (int)dim2, (int)dim3);
}
