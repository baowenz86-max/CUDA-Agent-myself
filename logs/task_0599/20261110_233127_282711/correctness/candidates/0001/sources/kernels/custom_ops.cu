// custom_ops.cu
// Element-wise absolute value and broadcast addition kernels.
// All launchers use raw pointers and cudaStream_t; no PyTorch dependencies.

#include <cuda_runtime.h>
#include <cstdint>

// Kernel for element-wise absolute value on float32 tensors.
// Supports arbitrary strides; index calculates each dimension's offset.
__global__ void abs_kernel_float(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel,
    int64_t ndim,
    const int64_t* __restrict__ out_strides,
    const int64_t* __restrict__ in_strides,
    const int64_t* __restrict__ shape)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < numel; linear += stride) {
        // Compute multi-dimensional indices from linear index
        int64_t rem = linear;
        int64_t out_offset = 0;
        int64_t in_offset = 0;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t coord = rem % shape[d];
            rem /= shape[d];
            out_offset += coord * out_strides[d];
            in_offset += coord * in_strides[d];
        }
        output[out_offset] = fabsf(input[in_offset]);
    }
}

// Kernel for broadcast addition: output[i] = x[i] + param[i % param_shape]
// Supports arbitrary strides for x and param. Param has fewer dims,
// broadcast over leading dimensions (batch).
__global__ void add_broadcast_kernel_float(
    float* __restrict__ output,
    const float* __restrict__ x,
    const float* __restrict__ param,
    int64_t numel,
    int64_t ndim,
    int64_t param_ndim,
    const int64_t* __restrict__ out_strides,
    const int64_t* __restrict__ x_strides,
    const int64_t* __restrict__ param_strides,
    const int64_t* __restrict__ shape)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < numel; linear += stride) {
        // Compute multi-dimensional indices from linear index
        int64_t rem = linear;
        int64_t out_offset = 0;
        int64_t x_offset = 0;
        int64_t param_offset = 0;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t coord = rem % shape[d];
            rem /= shape[d];
            out_offset += coord * out_strides[d];
            x_offset += coord * x_strides[d];
            // For param, broadcast: if d >= ndim - param_ndim, use param coordinate,
            // otherwise index 0 (since leading dims are broadcast).
            int64_t param_d = d - (ndim - param_ndim);
            if (param_d >= 0) {
                param_offset += coord * param_strides[param_d];
            }
        }
        output[out_offset] = x[x_offset] + param[param_offset];
    }
}

// Launch configuration helper: choose block size 256, calculate grid size with cap.
static inline int get_num_blocks(int64_t numel, int block_size) {
    int64_t blocks = (numel + block_size - 1) / block_size;
    // Cap grid to avoid excessive blocks, grid-stride loop handles remainder.
    const int64_t max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    return (int)blocks;
}

// Extern "C" launcher for abs_forward.
// Signature matches binding expectations.
extern "C" void abs_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    int64_t ndim,
    const int64_t* out_strides,
    const int64_t* in_strides,
    const int64_t* shape,
    cudaStream_t stream)
{
    if (numel == 0) return;
    int block_size = 256;
    int grid_size = get_num_blocks(numel, block_size);
    abs_kernel_float<<<grid_size, block_size, 0, stream>>>(
        output, input, numel, ndim, out_strides, in_strides, shape);
}

// Extern "C" launcher for add_scalar_tensor_forward.
extern "C" void add_scalar_tensor_forward_launcher(
    float* output,
    const float* x,
    const float* param,
    int64_t numel,
    int64_t ndim,
    int64_t param_ndim,
    const int64_t* out_strides,
    const int64_t* x_strides,
    const int64_t* param_strides,
    const int64_t* shape,
    cudaStream_t stream)
{
    if (numel == 0) return;
    int block_size = 256;
    int grid_size = get_num_blocks(numel, block_size);
    add_broadcast_kernel_float<<<grid_size, block_size, 0, stream>>>(
        output, x, param, numel, ndim, param_ndim,
        out_strides, x_strides, param_strides, shape);
}
