#include <cuda_runtime.h>
#include <cstdint>

// Kernel for max reduction over dimension 1 (channels) with keepdim semantics.
// Input tensor shape: [batch, C, H, W]
// Output values shape: [batch, 1, H, W]
// Output indices shape: [batch, 1, H, W] (int64)
// Supports arbitrary strides for input, values, and indices.
__global__ void max_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int batch,
    int C,
    int H,
    int W,
    long long input_stride_b,
    long long input_stride_c,
    long long input_stride_h,
    long long input_stride_w,
    long long out_stride_b,
    long long out_stride_h,
    long long out_stride_w)
{
    int64_t total = (int64_t)batch * H * W;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t linear = idx; linear < total; linear += stride) {
        int b = linear / (H * W);
        int rem = linear % (H * W);
        int h = rem / W;
        int w = rem % W;

        const float* in_base = input + b * input_stride_b + h * input_stride_h + w * input_stride_w;
        float max_val = in_base[0];
        int64_t max_idx = 0;

        for (int c = 1; c < C; ++c) {
            float val = in_base[c * input_stride_c];
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
        }

        float* out_val = values + b * out_stride_b + h * out_stride_h + w * out_stride_w;
        int64_t* out_idx = indices + b * out_stride_b + h * out_stride_h + w * out_stride_w;
        *out_val = max_val;
        *out_idx = max_idx;
    }
}

// Kernel for ReLU elementwise: output = max(input, 0)
// Handles arbitrary strides.
__global__ void relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total,
    const int64_t* __restrict__ input_strides,  // array of strides for each dimension (ndim)
    const int64_t* __restrict__ output_strides,
    int ndim)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t linear = idx; linear < total; linear += stride) {
        int64_t remaining = linear;
        int64_t in_offset = 0;
        int64_t out_offset = 0;
        // Compute multidimensional indices from linear index using output shape
        // We assume shapes for input and output are identical.
        // Since we don't have shapes here, we use strides and total numel.
        // This trick works only if we assume the tensor has no zero dimensions
        // and is contiguous in the sense that linear index maps to offset via strides.
        // To be fully general, we would need shape information; however, the launcher
        // will pass pointer to shape and we can use the same approach as max kernel.
        // But to keep simple and correct for the given task, we will use a different approach:
        // For ReLU, we can treat input and output as 4D tensors of shape [batch, 1, H, W]
        // because in this specific usage, the input to relu is the values from max_dim1,
        // which has shape [batch, 1, H, W] and is contiguous (as created by torch.empty_like).
        // We will implement a specialized 4D ReLU kernel for this case.
        // However, to maintain generality, we could write a generic ndim kernel with shape.
        // Given the contract, only this specific shape is used, so we implement a 4D version.
        // Thus, we remove the generic version and use a specialized one below.
    }
}

// Specialized ReLU kernel for 4D tensors of shape [batch, 1, H, W] with arbitrary strides.
__global__ void relu_4d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int H,
    int W,
    long long input_stride_b,
    long long input_stride_c,  // unused but for completeness
    long long input_stride_h,
    long long input_stride_w,
    long long output_stride_b,
    long long output_stride_c,
    long long output_stride_h,
    long long output_stride_w)
{
    int64_t total = (int64_t)batch * H * W;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t linear = idx; linear < total; linear += stride) {
        int b = linear / (H * W);
        int rem = linear % (H * W);
        int h = rem / W;
        int w = rem % W;

        const float* in_ptr = input + b * input_stride_b + h * input_stride_h + w * input_stride_w;
        float* out_ptr = output + b * output_stride_b + h * output_stride_h + w * output_stride_w;
        float val = *in_ptr;
        *out_ptr = val > 0.0f ? val : 0.0f;
    }
}

// Launcher for custom_max_dim1.
// Input: x (float32 tensor, CUDA) shape [batch, C, H, W], arbitrary strides.
// Output: values (float32) and indices (int64) shape [batch, 1, H, W], contiguous.
// This launcher is called from the binding wrapper (to be implemented in C++).
extern "C" void custom_max_dim1_launcher(
    const float* x,
    float* values,
    int64_t* indices,
    int batch,
    int C,
    int H,
    int W,
    long long x_stride_b,
    long long x_stride_c,
    long long x_stride_h,
    long long x_stride_w,
    long long out_stride_b,
    long long out_stride_h,
    long long out_stride_w,
    cudaStream_t stream)
{
    int64_t total_elements = (int64_t)batch * H * W;
    int block_size = 256;
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535; // grid-stride loop handles rest

    max_dim1_kernel<<<grid_size, block_size, 0, stream>>>(
        x, values, indices,
        batch, C, H, W,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        out_stride_b, out_stride_h, out_stride_w);
}

// Launcher for custom_relu.
// Input: x (float32 tensor, CUDA) shape [batch, 1, H, W], arbitrary strides.
// Output: float32 tensor same shape, contiguous.
extern "C" void custom_relu_launcher(
    const float* x,
    float* output,
    int batch,
    int H,
    int W,
    long long x_stride_b,
    long long x_stride_c,
    long long x_stride_h,
    long long x_stride_w,
    long long out_stride_b,
    long long out_stride_c,
    long long out_stride_h,
    long long out_stride_w,
    cudaStream_t stream)
{
    int64_t total_elements = (int64_t)batch * H * W;
    int block_size = 256;
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    if (grid_size > 65535) grid_size = 65535;

    relu_4d_kernel<<<grid_size, block_size, 0, stream>>>(
        x, output,
        batch, H, W,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w);
}
