#include <cuda_runtime.h>
#include <float.h>

// Softmax along dimension 1: input shape [N, C, L]
// Each (n, l) position computes softmax over c = 0..C-1.
// x_ptr: input data, may have arbitrary strides
// y_ptr: output data, same strides as input or compact? The contract says returns new tensor, possibly same strides as input? Actually we will produce contiguous output? PyTorch softmax returns contiguous? Likely contiguous. But launcher receives output pointer and strides; binding will allocate properly.
// We'll assume output is contiguous with same strides as a contiguous [N,C,L] tensor, but we can pass strides as parameters.
// To support any strides, we pass input_strides (3 values) and output_strides (3 values), plus dimensions N,C,L.
__global__ void softmax_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N, int C, int L,
    long long in_stride_n, long long in_stride_c, long long in_stride_l,
    long long out_stride_n, long long out_stride_c, long long out_stride_l)
{
    // Total number of independent softmax problems: N * L
    long long total = (long long)N * L;
    // grid-stride loop over all problems
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (long long)blockDim.x * gridDim.x)
    {
        int n = idx / L;
        int l = idx - (long long)n * L;
        // compute max over c
        float max_val = -FLT_MAX;
        for (int c = 0; c < C; ++c) {
            long long in_idx = (long long)n * in_stride_n + (long long)c * in_stride_c + (long long)l * in_stride_l;
            float val = x[in_idx];
            if (val > max_val) max_val = val;
        }
        // compute sum exp
        float sum_exp = 0.0f;
        for (int c = 0; c < C; ++c) {
            long long in_idx = (long long)n * in_stride_n + (long long)c * in_stride_c + (long long)l * in_stride_l;
            float val = __expf(x[in_idx] - max_val);
            sum_exp += val;
            // Optionally store exp in a temporary? Not needed, recompute later.
        }
        // write output
        float inv_sum = 1.0f / sum_exp;
        for (int c = 0; c < C; ++c) {
            long long in_idx = (long long)n * in_stride_n + (long long)c * in_stride_c + (long long)l * in_stride_l;
            long long out_idx = (long long)n * out_stride_n + (long long)c * out_stride_c + (long long)l * out_stride_l;
            float val = __expf(x[in_idx] - max_val) * inv_sum;
            y[out_idx] = val;
        }
    }
}

// Launcher for softmax along dim1. Accepts raw pointers and strides.
// The binding should allocate output and pass strides accordingly.
// We assume input and output are both float32 CUDA tensors with 3 dimensions.
// No error checking here, binding does that.
extern "C" void softmax_dim1_launcher(
    const float* input,
    float* output,
    int N, int C, int L,
    long long in_stride_n, long long in_stride_c, long long in_stride_l,
    long long out_stride_n, long long out_stride_c, long long out_stride_l,
    cudaStream_t stream)
{
    // total number of independent problems
    long long total = (long long)N * L;
    // choose block size and grid size
    int block_size = 256;
    long long grid_size_long = (total + block_size - 1) / block_size;
    // cap grid to reasonable max (e.g., 65535) and use grid-stride loop (kernel already has loop)
    int grid_size = (int)min(grid_size_long, 65535LL);
    softmax_dim1_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, N, C, L,
        in_stride_n, in_stride_c, in_stride_l,
        out_stride_n, out_stride_c, out_stride_l);
}

// Note: The convolution operation (conv_transpose1d_ext) will be implemented using cuDNN
// in a separate C++ binding file (kernels/*_binding.cpp) as per instructions. This .cu file
// contains only the softmax kernel. The convolution launcher will be declared extern and
// linked from the C++ file.
