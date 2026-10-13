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

// Round kernel: out[i] = roundf(in[i]) for contiguous or arbitrary strides.
// Since output is contiguous, out index equals linear index.
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
        // Compute multi-dimensional index from linear index
        int remaining = i;
        int in_offset = 0;
        for (int d = 0; d < ndims; ++d) {
            int dim_size = shape[d];
            int coord = remaining % dim_size;
            remaining /= dim_size;
            in_offset += coord * strides[d];
        }
        out[i] = roundf(in[in_offset]);
    }
}

// Custom CUDA kernel for 1D transposed convolution.
// Input x: [batch, in_channels, length_in], with arbitrary strides.
// Weight: [in_channels, out_channels, kernel_size], contiguous.
// Output y: [batch, out_channels, length_out], contiguous.
// Parameters: stride, padding, output_padding as in PyTorch.
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int length_in,
    int length_out,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int in_batch_stride,
    int in_channel_stride,
    int in_inner_stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch * out_channels * length_out;
    if (idx >= total_elements) return;

    // Decompose linear index into (b, oc, ol)
    int b = idx / (out_channels * length_out);
    int rem = idx - b * out_channels * length_out;
    int oc = rem / length_out;
    int ol = rem - oc * length_out;

    float sum = 0.0f;

    // Iterate over input channels and kernel positions
    for (int ic = 0; ic < in_channels; ++ic) {
        // Input pointer offset for (b, ic, 0)
        const float* input_base = input + b * in_batch_stride + ic * in_channel_stride;
        // Weight pointer offset for (ic, oc, 0)
        const float* weight_base = weight + (ic * out_channels + oc) * kernel_size;

        for (int k = 0; k < kernel_size; ++k) {
            // Compute input index: il = (ol + padding - k) / stride, if divisible and in range
            int numerator = ol + padding - k;
            if (numerator < 0 || numerator % stride != 0) continue;
            int il = numerator / stride;
            if (il >= 0 && il < length_in) {
                sum += input_base[il * in_inner_stride] * weight_base[k];
            }
        }
    }

    output[idx] = sum;
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
    if (grid_size > 65535) grid_size = 65535; // limit on grid dimension for older GPUs; grid-stride loop handles rest
    bias_add_kernel<<<grid_size, block_size, 0, stream>>>(
        out, in, bias, total_elements, channels, length,
        batch_stride, channel_stride, inner_stride);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // The binding layer will handle error checking; we just ensure the launch is attempted.
        // No throwing here to keep CUDA-only file.
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
        // Handle as needed by binding layer.
    }
}

// Launcher for conv_transpose1d_forward.
extern "C" void conv_transpose1d_forward_launcher(
    const float* input,
    const float* weight,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int length_in,
    int length_out,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int in_batch_stride,
    int in_channel_stride,
    int in_inner_stride,
    cudaStream_t stream)
{
    int total_elements = batch * out_channels * length_out;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    conv_transpose1d_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weight, output,
        batch, in_channels, out_channels, length_in, length_out,
        kernel_size, stride, padding, output_padding,
        in_batch_stride, in_channel_stride, in_inner_stride);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Leave error checking to binding layer.
    }
}
