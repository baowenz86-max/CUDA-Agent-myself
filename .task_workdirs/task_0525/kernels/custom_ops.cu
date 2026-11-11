#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel for computing variance and mean along a specified dimension.
// Input: x (float), output var (float), output mean (float).
// Parameters:
//   x: input tensor data pointer (float*)
//   var_out: output variance tensor data pointer (float*)
//   mean_out: output mean tensor data pointer (float*)
//   outer_size: number of slices along dim (product of dimensions before dim)
//   inner_size: number of elements per slice along dim (dim size)
//   stride_x_outer: stride of x for outer dimension (in elements)
//   stride_x_inner: stride of x for inner dimension (usually 1, but could be different)
//   stride_out_outer: stride of output for outer dimension (in elements)
__global__ void var_mean_kernel(
    const float* __restrict__ x,
    float* __restrict__ var_out,
    float* __restrict__ mean_out,
    int64_t outer_size,
    int64_t inner_size,
    int64_t stride_x_outer,
    int64_t stride_x_inner,
    int64_t stride_out_outer)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t o = idx; o < outer_size; o += stride) {
        const float* slice = x + o * stride_x_outer;
        double sum = 0.0;
        double sum_sq = 0.0;

        for (int64_t i = 0; i < inner_size; ++i) {
            float val = slice[i * stride_x_inner];
            sum += val;
            sum_sq += val * val;
        }

        double mean = sum / inner_size;
        // Unbiased variance (correction=1): sum((x-mean)^2) / (n-1)
        double var = (sum_sq - sum * mean) / (inner_size - 1);
        // Numerical stability alternative: two-pass, but one-pass is acceptable with double accumulation;
        // Torch uses two-pass? Actually torch.var_mean uses Welford? For correctness we can use two-pass to match better.
        // Let's compute variance with two-pass to match PyTorch more closely.
        // Recompute: var = sum((x-mean)^2)/(n-1)
        double var2 = 0.0;
        for (int64_t i = 0; i < inner_size; ++i) {
            float val = slice[i * stride_x_inner];
            double diff = val - mean;
            var2 += diff * diff;
        }
        var = var2 / (inner_size - 1);

        int64_t out_idx = o * stride_out_outer;
        var_out[out_idx] = (float)var;
        mean_out[out_idx] = (float)mean;
    }
}

// Launcher for var_mean kernel. All parameters are raw pointers and scalars.
// The output pointers should point to pre-allocated tensors with shape determined by keepdim.
extern "C" void var_mean_dim_launcher(
    const float* x,
    float* var_out,
    float* mean_out,
    int64_t outer_size,
    int64_t inner_size,
    int64_t stride_x_outer,
    int64_t stride_x_inner,
    int64_t stride_out_outer,
    cudaStream_t stream)
{
    // Use 256 threads per block, grid sized to cover all outer elements.
    int threads = 256;
    int blocks = (int)((outer_size + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid limit for x dimension
    // For very large outer_size (unlikely here), use grid-stride loop with max grid.
    // Our kernel already has grid-stride loop, so a smaller grid is fine.
    // But we want to ensure adequate parallelism; for outer_size = 2M, blocks=8192 < 65535, OK.
    var_mean_kernel<<<blocks, threads, 0, stream>>>(
        x, var_out, mean_out,
        outer_size, inner_size,
        stride_x_outer, stride_x_inner,
        stride_out_outer);
}
