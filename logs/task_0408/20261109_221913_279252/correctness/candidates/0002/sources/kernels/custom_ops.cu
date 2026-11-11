#include <cuda_runtime.h>
#include <cfloat>

// Helper to compute linear index from multi-index given strides
__device__ __forceinline__ int64_t compute_offset(const int64_t* indices, const int64_t* strides, int ndim) {
    int64_t offset = 0;
    for (int i = 0; i < ndim; ++i) {
        offset += indices[i] * strides[i];
    }
    return offset;
}

// Kernel: compute max along dimension dim, keepdim=True.
// Each thread computes one or more output elements using grid-stride loop.
template <typename scalar_t>
__global__ void reduce_max_4d_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int64_t* __restrict__ dims,
    const int64_t* __restrict__ strides,
    int dim,
    int64_t output_size)
{
    int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

    int64_t indices[4];
    int64_t dim_size = dims[dim];

    for (int64_t i = out_idx; i < output_size; i += total_threads) {
        // Decode flattened output index into multi-index (with dim set to 0)
        int64_t remainder = i;
        for (int d = 3; d >= 0; --d) {
            if (d == dim) {
                indices[d] = 0;
            } else {
                indices[d] = remainder % dims[d];
                remainder /= dims[d];
            }
        }

        // Base offset for reduction dimension index 0
        int64_t base = compute_offset(indices, strides, 4);

        scalar_t max_val = -FLT_MAX;
        for (int64_t k = 0; k < dim_size; ++k) {
            indices[dim] = k;
            int64_t offset = compute_offset(indices, strides, 4);
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
    const int64_t* __restrict__ dims,
    const int64_t* __restrict__ strides,
    int dim,
    int64_t output_size)
{
    int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_threads = static_cast<int64_t>(gridDim.x) * blockDim.x;

    int64_t indices[4];
    int64_t dim_size = dims[dim];

    for (int64_t i = out_idx; i < output_size; i += total_threads) {
        int64_t remainder = i;
        for (int d = 3; d >= 0; --d) {
            if (d == dim) {
                indices[d] = 0;
            } else {
                indices[d] = remainder % dims[d];
                remainder /= dims[d];
            }
        }

        int64_t base = compute_offset(indices, strides, 4);
        scalar_t prod_val = static_cast<scalar_t>(1.0);
        for (int64_t k = 0; k < dim_size; ++k) {
            indices[dim] = k;
            int64_t offset = compute_offset(indices, strides, 4);
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
    const int64_t* x_dims,
    const int64_t* x_strides,
    cudaStream_t stream)
{
    // Compute output_size = product of dims except dim (since dim size becomes 1)
    int64_t output_size = 1;
    for (int d = 0; d < 4; ++d) {
        if (d != dim) {
            output_size *= x_dims[d];
        }
    }

    int threads = 128;
    int blocks = (output_size + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    reduce_max_4d_kernel<float><<<blocks, threads, 0, stream>>>(
        input, output, x_dims, x_strides, (int)dim, output_size);
}

// Launcher for product reduction.
extern "C" void reduce_prod_4d_launcher(
    const float* input,
    float* output,
    int64_t dim,
    const int64_t* x_dims,
    const int64_t* x_strides,
    cudaStream_t stream)
{
    int64_t output_size = 1;
    for (int d = 0; d < 4; ++d) {
        if (d != dim) {
            output_size *= x_dims[d];
        }
    }

    int threads = 128;
    int blocks = (output_size + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;

    reduce_prod_4d_kernel<float><<<blocks, threads, 0, stream>>>(
        input, output, x_dims, x_strides, (int)dim, output_size);
}
