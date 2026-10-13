#include <cuda_runtime.h>

// Bias add kernel: out[b,c,l] = x[b,c,l] + bias[c]
// Supports arbitrary strides.
__global__ void bias_add_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    const float* __restrict__ bias,
    int total_elements,
    int channels,
    int inner_size,
    int batch_stride,
    int channel_stride,
    int inner_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_elements; i += stride) {
        int b = i / (channels * inner_size);
        int rem = i - b * channels * inner_size;
        int c = rem / inner_size;
        int l = rem - c * inner_size;
        int in_offset = b * batch_stride + c * channel_stride + l * inner_stride;
        out[i] = in[in_offset] + bias[c];
    }
}

// Round kernel: out[i] = rintf(in[i]) (matches torch.round, half to even with sign preservation)
__global__ void round_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int total_elements,
    int ndims,
    const int* __restrict__ shape,
    const int* __restrict__ strides)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total_elements; i += stride) {
        int remaining = i;
        int in_offset = 0;
        for (int d = 0; d < ndims; ++d) {
            int dim_size = shape[d];
            int coord = remaining % dim_size;
            remaining /= dim_size;
            in_offset += coord * strides[d];
        }
        out[i] = rintf(in[in_offset]);
    }
}

// Launcher for bias_add_forward.
extern "C" void bias_add_forward_launcher(
    float* out,
    const float* in,
    const float* bias,
    int batch,
    int channels,
    int length,
    int batch_stride,
    int channel_stride,
    int inner_stride,
    cudaStream_t stream)
{
    int total_elements = batch * channels * length;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    bias_add_kernel<<<grid_size, block_size, 0, stream>>>(
        out, in, bias, total_elements, channels, length,
        batch_stride, channel_stride, inner_stride);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // No throwing; binding will check.
    }
}

// Launcher for round_elementwise_forward.
extern "C" void round_elementwise_forward_launcher(
    float* out,
    const float* in,
    int ndims,
    const int* shape,
    const int* strides,
    int total_elements,
    cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    round_kernel<<<grid_size, block_size, 0, stream>>>(
        out, in, total_elements, ndims, shape, strides);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Binding will check.
    }
}
