// kernels/custom_ops.cu
// CUDA kernels for max_reduce and relu6 used in model_new.py
// All launchers have C linkage for binding.cpp

#include <cuda_runtime.h>
#include <cstdint>

// ---------------- max_reduce over a specified dimension ----------------

// Specialized kernel for dim=1, keepdim=true, contiguous 4D input [N,C,H,W]
__global__ void max_reduce_special(
    const float* __restrict__ input,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int N, int C, int H, int W)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)N * H * W;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        int n = idx / (H * W);
        int rem = idx - n * (H * W);
        int h = rem / W;
        int w = rem - h * W;

        const float* base = input + ((int64_t)n * C + 0) * H * W + h * W + w;
        float max_val = base[0];
        int64_t max_idx = 0;
        for (int c = 1; c < C; ++c) {
            float val = base[c * H * W];
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
        }
        int64_t out_idx = (int64_t)n * H * W + h * W + w;
        values[out_idx] = max_val;
        indices[out_idx] = max_idx;
    }
}

// Generic kernel for other cases
__global__ void max_reduce_generic(
    const float* __restrict__ input,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int64_t reduced_size,
    int64_t outer_size,
    int64_t inner_size,
    int64_t input_stride_reduced,
    int64_t input_stride_outer,
    int64_t input_stride_inner,
    int64_t output_stride_outer,
    int64_t output_stride_inner)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = outer_size * inner_size;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        int64_t outer = idx / inner_size;
        int64_t inner = idx - outer * inner_size;
        const float* base = input + outer * input_stride_outer + inner * input_stride_inner;
        float max_val = base[0];
        int64_t max_idx = 0;
        for (int64_t r = 1; r < reduced_size; ++r) {
            float val = base[r * input_stride_reduced];
            if (val > max_val) {
                max_val = val;
                max_idx = r;
            }
        }
        values[outer * output_stride_outer + inner * output_stride_inner] = max_val;
        indices[outer * output_stride_outer + inner * output_stride_inner] = max_idx;
    }
}

// Launcher for max_reduce
extern "C" void max_reduce_launcher(
    const float* x,
    float* out_values,
    int64_t* out_indices,
    int ndim,
    const int64_t* input_shape,
    const int64_t* input_strides,
    const int64_t* output_shape,
    const int64_t* output_strides,
    int64_t dim,
    bool keepdim,
    cudaStream_t stream)
{
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) return;

    int64_t outer_size = 1;
    for (int i = 0; i < dim; ++i) outer_size *= input_shape[i];
    int64_t inner_size = 1;
    for (int i = dim+1; i < ndim; ++i) inner_size *= input_shape[i];
    int64_t reduced_size = input_shape[dim];

    bool is_special = (ndim == 4 && dim == 1 && keepdim &&
                       input_strides[3] == 1 && input_strides[2] == input_shape[3] &&
                       input_strides[1] == input_shape[2]*input_shape[3] &&
                       input_strides[0] == input_shape[1]*input_shape[2]*input_shape[3]);
    if (is_special) {
        int N = input_shape[0];
        int C = input_shape[1];
        int H = input_shape[2];
        int W = input_shape[3];
        int64_t total = (int64_t)N * H * W;
        int block = 256;
        int grid = (int)((total + block - 1) / block);
        max_reduce_special<<<grid, block, 0, stream>>>(x, out_values, out_indices, N, C, H, W);
    } else {
        int64_t input_stride_reduced = 1;
        for (int i = dim+1; i < ndim; ++i) input_stride_reduced *= input_shape[i];
        int64_t input_stride_outer = input_stride_reduced * reduced_size;
        int64_t input_stride_inner = 1;
        int64_t output_stride_outer = inner_size;
        int64_t output_stride_inner = 1;
        int64_t total = outer_size * inner_size;
        int block = 256;
        int grid = (int)((total + block - 1) / block);
        max_reduce_generic<<<grid, block, 0, stream>>>(
            x, out_values, out_indices, reduced_size, outer_size, inner_size,
            input_stride_reduced, input_stride_outer, input_stride_inner,
            output_stride_outer, output_stride_inner);
    }
}

// ---------------- relu6 ----------------
__global__ void relu6_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t n)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        float v = input[idx];
        v = v > 0.0f ? v : 0.0f;
        v = v < 6.0f ? v : 6.0f;
        output[idx] = v;
    }
}

extern "C" void relu6_launcher(const float* input, float* output, int64_t n, cudaStream_t stream)
{
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    relu6_kernel<<<grid, block, 0, stream>>>(input, output, n);
}
