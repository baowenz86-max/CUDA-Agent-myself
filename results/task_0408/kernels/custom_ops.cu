#include <cuda_runtime.h>
#include <cfloat>

// Helper to compute linear offset from multi-index given strides
__device__ __forceinline__ int64_t compute_offset(
    int64_t i0, int64_t i1, int64_t i2, int64_t i3,
    int64_t s0, int64_t s1, int64_t s2, int64_t s3)
{
    return i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3;
}

// Kernel: compute max along dimension dim, keepdim=True.
template <typename scalar_t>
__global__ void reduce_max_4d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim,
    int64_t d0, int64_t d1, int64_t d2, int64_t d3,
    int64_t s0, int64_t s1, int64_t s2, int64_t s3,
    int64_t output_size)
{
    int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

    int64_t dim_size;
    int64_t dim_stride;
    // Determine reduction dimension size and stride
    if (dim == 0) { dim_size = d0; dim_stride = s0; }
    else if (dim == 1) { dim_size = d1; dim_stride = s1; }
    else if (dim == 2) { dim_size = d2; dim_stride = s2; }
    else { dim_size = d3; dim_stride = s3; }

    for (int64_t i = out_idx; i < output_size; i += total_threads) {
        // Decode flattened output index into multi-index with dim set to 0
        int64_t remainder = i;
        int64_t idx[4] = {0, 0, 0, 0};
        // The output has dimension sizes: all dims except reduction dim
        // Loop over dimensions from 3 down to 0, skipping reduction dim
        for (int d = 3; d >= 0; --d) {
            if (d == dim) {
                idx[d] = 0;
            } else {
                int64_t cur_dim = (d == 0) ? d0 : (d == 1) ? d1 : (d == 2) ? d2 : d3;
                idx[d] = remainder % cur_dim;
                remainder /= cur_dim;
            }
        }

        // Base offset with reduction index 0
        int64_t base = compute_offset(idx[0], idx[1], idx[2], idx[3], s0, s1, s2, s3);

        scalar_t max_val = -FLT_MAX;
        for (int64_t k = 0; k < dim_size; ++k) {
            int64_t offset = base + k * dim_stride;
            scalar_t val = input[offset];
            if (val > max_val) {
                max_val = val;
            }
        }
        output[i] = max_val;
    }
}

// Kernel: compute product along dimension dim, keepdim=True.
template <typename scalar_t>
__global__ void reduce_prod_4d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int dim,
    int64_t d0, int64_t d1, int64_t d2, int64_t d3,
    int64_t s0, int64_t s1, int64_t s2, int64_t s3,
    int64_t output_size)
{
    int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

    int64_t dim_size;
    int64_t dim_stride;
    if (dim == 0) { dim_size = d0; dim_stride = s0; }
    else if (dim == 1) { dim_size = d1; dim_stride = s1; }
    else if (dim == 2) { dim_size = d2; dim_stride = s2; }
    else { dim_size = d3; dim_stride = s3; }

    for (int64_t i = out_idx; i < output_size; i += total_threads) {
        int64_t remainder = i;
        int64_t idx[4] = {0, 0, 0, 0};
        for (int d = 3; d >= 0; --d) {
            if (d == dim) {
                idx[d] = 0;
            } else {
                int64_t cur_dim = (d == 0) ? d0 : (d == 1) ? d1 : (d == 2) ? d2 : d3;
                idx[d] = remainder % cur_dim;
                remainder /= cur_dim;
            }
        }

        int64_t base = compute_offset(idx[0], idx[1], idx[2], idx[3], s0, s1, s2, s3);
        scalar_t prod_val = static_cast<scalar_t>(1.0);
        for (int64_t k = 0; k < dim_size; ++k) {
            int64_t offset = base + k * dim_stride;
            prod_val *= input[offset];
        }
        output[i] = prod_val;
    }
}

// Launcher for max reduction.
extern "C" void reduce_max_4d_launcher(
    const float* input,
    float* output,
    int64_t dim,
    int64_t d0, int64_t d1, int64_t d2, int64_t d3,
    int64_t s0, int64_t s1, int64_t s2, int64_t s3,
    cudaStream_t stream)
{
    int64_t output_size = 1;
    if (dim != 0) output_size *= d0;
    if (dim != 1) output_size *= d1;
    if (dim != 2) output_size *= d2;
    if (dim != 3) output_size *= d3;

    int threads = 128;
    int blocks = (output_size + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    reduce_max_4d_kernel<float><<<blocks, threads, 0, stream>>>(
        input, output, (int)dim,
        d0, d1, d2, d3,
        s0, s1, s2, s3,
        output_size);
}

// Launcher for product reduction.
extern "C" void reduce_prod_4d_launcher(
    const float* input,
    float* output,
    int64_t dim,
    int64_t d0, int64_t d1, int64_t d2, int64_t d3,
    int64_t s0, int64_t s1, int64_t s2, int64_t s3,
    cudaStream_t stream)
{
    int64_t output_size = 1;
    if (dim != 0) output_size *= d0;
    if (dim != 1) output_size *= d1;
    if (dim != 2) output_size *= d2;
    if (dim != 3) output_size *= d3;

    int threads = 128;
    int blocks = (output_size + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    reduce_prod_4d_kernel<float><<<blocks, threads, 0, stream>>>(
        input, output, (int)dim,
        d0, d1, d2, d3,
        s0, s1, s2, s3,
        output_size);
}
