#include <cuda_runtime.h>

// Elementwise sigmoid: y = 1 / (1 + exp(-x))
__global__ void sigmoid_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long numel,
    long long* __restrict__ in_strides,
    long long* __restrict__ out_strides,
    int ndim)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long linear = idx; linear < numel; linear += stride) {
        // compute multi-dimensional index from linear index
        long long in_offset = 0;
        long long out_offset = 0;
        long long tmp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
            long long coord = tmp % in_strides[d] == 0 ? 0 : 0; // placeholder no
            // actual computation below
        }
        // Correct implementation:
        long long rem = linear;
        for (int d = 0; d < ndim; ++d) {
            long long size_d = in_strides[d] == 0 ? 1 : 0; // won't be used
        }
        // Since strides may be zero for size-1 dims, we need sizes. But we can compute using original shape.
        // We only have strides, not sizes. The launcher can pass sizes as well.
    }
}

// Actually easier: pass integer sizes to compute linear index mapping.
// Let's define a more general elementwise kernel with strides and sizes.
__global__ void sigmoid_kernel_v2(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long numel,
    const long long* __restrict__ input_strides,
    const long long* __restrict__ output_strides,
    const long long* __restrict__ sizes,
    int ndim)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;
    for (long long linear = idx; linear < numel; linear += step) {
        long long in_offset = 0;
        long long out_offset = 0;
        long long tmp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
            long long coord = tmp % sizes[d];
            tmp /= sizes[d];
            in_offset += coord * input_strides[d];
            out_offset += coord * output_strides[d];
        }
        float x = input[in_offset];
        // stable sigmoid
        float exp_val = __expf(-x);
        float y = 1.0f / (1.0f + exp_val);
        output[out_offset] = y;
    }
}

// multiply scalar kernel
__global__ void multiply_scalar_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long numel,
    float scalar,
    const long long* __restrict__ input_strides,
    const long long* __restrict__ output_strides,
    const long long* __restrict__ sizes,
    int ndim)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;
    for (long long linear = idx; linear < numel; linear += step) {
        long long in_offset = 0;
        long long out_offset = 0;
        long long tmp = linear;
        for (int d = ndim - 1; d >= 0; --d) {
            long long coord = tmp % sizes[d];
            tmp /= sizes[d];
            in_offset += coord * input_strides[d];
            out_offset += coord * output_strides[d];
        }
        output[out_offset] = input[in_offset] * scalar;
    }
}

// Launcher for sigmoid: output tensor with same shape, but may have different strides (contiguous allocation)
// We assume output is contiguous; we can also pass output strides if needed.
extern "C" void custom_sigmoid_launcher(
    float* output,
    const float* input,
    long long numel,
    const long long* input_strides,
    const long long* output_strides,
    const long long* sizes,
    int ndim,
    cudaStream_t stream)
{
    int block_size = 256;
    long long grid_size_long = (numel + block_size - 1) / block_size;
    if (grid_size_long > 65535) grid_size_long = 65535;
    dim3 grid((unsigned)grid_size_long);
    sigmoid_kernel_v2<<<grid, block_size, 0, stream>>>(
        output, input, numel, input_strides, output_strides, sizes, ndim);
}

extern "C" void custom_multiply_scalar_launcher(
    float* output,
    const float* input,
    long long numel,
    float scalar,
    const long long* input_strides,
    const long long* output_strides,
    const long long* sizes,
    int ndim,
    cudaStream_t stream)
{
    int block_size = 256;
    long long grid_size_long = (numel + block_size - 1) / block_size;
    if (grid_size_long > 65535) grid_size_long = 65535;
    dim3 grid((unsigned)grid_size_long);
    multiply_scalar_kernel<<<grid, block_size, 0, stream>>>(
        output, input, numel, scalar, input_strides, output_strides, sizes, ndim);
}
