#include <cuda_runtime.h>

// BatchNorm1d forward kernel (eval mode)
// Computes: y[n,c,l] = (input[n,c,l] - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
// If weight is null: omit weight multiplication (default 1.0)
// If bias is null: omit bias addition (default 0.0)
// Supports non-contiguous input via explicit strides.
__global__ void batch_norm_1d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C, int L,
    long long input_stride_n,   // stride in bytes? no, we will pass element strides from launcher
    long long input_stride_c,
    long long input_stride_l,
    long long output_stride_n,
    long long output_stride_c,
    long long output_stride_l,
    float eps
) {
    // Grid-stride loop over flat index of output elements, assuming output shape [N,C,L]
    int total = N * C * L;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += gridDim.x * blockDim.x) {
        // decompose idx into (n,c,l)
        int n = idx / (C * L);
        int rem = idx - n * (C * L);
        int c = rem / L;
        int l = rem - c * L;

        // compute input and output addresses using strides
        const float* in_ptr = input + n * input_stride_n + c * input_stride_c + l * input_stride_l;
        float* out_ptr = output + n * output_stride_n + c * output_stride_c + l * output_stride_l;

        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float val = (*in_ptr - mean) * inv_std;
        if (weight != nullptr) {
            val *= weight[c];
        }
        if (bias != nullptr) {
            val += bias[c];
        }
        *out_ptr = val;
    }
}

// Launcher (C interface, no PyTorch dependencies)
// Parameters:
//   input, output: pointers to float arrays
//   running_mean, running_var: valid pointers (size C)
//   weight, bias: may be nullptr, size C if non-null
//   N, C, L: dimensions
//   strides: element counts between consecutive elements in each dimension
//   eps: epsilon value (float, passed as float; contract uses double but converts)
//   stream: CUDA stream to launch on
extern "C" void batch_norm_1d_forward_launcher(
    const float* input,
    float* output,
    const float* running_mean,
    const float* running_var,
    const float* weight,
    const float* bias,
    int N, int C, int L,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_l,
    long long output_stride_n,
    long long output_stride_c,
    long long output_stride_l,
    float eps,
    cudaStream_t stream
) {
    int total = N * C * L;
    if (total <= 0) return;

    // Choose block size and grid size for good occupancy
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Cap blocks to a reasonable number to avoid excessive grid launch overhead; grid-stride loop handles rest
    const int max_blocks = 65535; // safe for all GPUs
    if (blocks > max_blocks) blocks = max_blocks;

    batch_norm_1d_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, output, running_mean, running_var, weight, bias,
        N, C, L,
        input_stride_n, input_stride_c, input_stride_l,
        output_stride_n, output_stride_c, output_stride_l,
        eps
    );
}
