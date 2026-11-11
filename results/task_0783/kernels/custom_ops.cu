#include <cuda_runtime.h>
#include <cstdint>

// Kernel: masked fill with threshold
// output(i) = (input(i) < mask_value) ? 0 : input(i)
// Element indexing based on flat contiguous output index, but need to map to input using strides.
// We assume output has same strides as input for masked_fill (but output may be allocated as empty_like which preserves strides).
// To support arbitrary strides, we compute 4D indices from flat index assuming a fixed shape, then use strides.
// Since shape is 4D [B, C, H, W], we can compute indices using division/modulo.
__global__ void masked_fill_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t total_elements,
    float mask_value,
    int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t stride_b, int64_t stride_c, int64_t stride_h, int64_t stride_w
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // Compute multi-dimensional index from flat idx in row-major (B,C,H,W)
    int64_t tmp = idx;
    int64_t w = tmp % W;
    tmp /= W;
    int64_t h = tmp % H;
    tmp /= H;
    int64_t c = tmp % C;
    tmp /= C;
    int64_t b = tmp;

    int64_t offset = b * stride_b + c * stride_c + h * stride_h + w * stride_w;
    float val = input[offset];
    output[offset] = (val < mask_value) ? 0.0f : val;
}

// Kernel: concatenate two tensors along dimension 1 (channels)
// Output is contiguous with shape [B, C1+C2, H, W].
// Inputs x and y may have their own strides, but output is contiguous.
// We copy element from x into first C1 channels, y into next C2 channels.
__global__ void cat_dim1_kernel(
    float* __restrict__ output,
    const float* __restrict__ x,
    const float* __restrict__ y,
    int64_t total_output_elements,
    int64_t B, int64_t C1, int64_t C2, int64_t H, int64_t W,
    int64_t x_stride_b, int64_t x_stride_c, int64_t x_stride_h, int64_t x_stride_w,
    int64_t y_stride_b, int64_t y_stride_c, int64_t y_stride_h, int64_t y_stride_w
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_output_elements) return;

    // Output contiguous layout
    int64_t tmp = idx;
    int64_t w = tmp % W;
    tmp /= W;
    int64_t h = tmp % H;
    tmp /= H;
    int64_t c_out = tmp % (C1 + C2);
    tmp /= (C1 + C2);
    int64_t b = tmp;

    int64_t output_offset = idx; // already contiguous flat index

    if (c_out < C1) {
        // from x
        int64_t c = c_out;
        int64_t input_offset = b * x_stride_b + c * x_stride_c + h * x_stride_h + w * x_stride_w;
        output[output_offset] = x[input_offset];
    } else {
        // from y
        int64_t c = c_out - C1;
        int64_t input_offset = b * y_stride_b + c * y_stride_c + h * y_stride_h + w * y_stride_w;
        output[output_offset] = y[input_offset];
    }
}

// Host launcher for custom_masked_fill
// Parameters:
//   output: pointer to output float tensor (same shape/strides as input)
//   input: pointer to input float tensor
//   total_elements: number of elements
//   mask_value: scalar threshold
//   shapes and strides
//   stream: cudaStream_t
void launch_masked_fill(
    float* output,
    const float* input,
    int64_t total_elements,
    float mask_value,
    int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t stride_b, int64_t stride_c, int64_t stride_h, int64_t stride_w,
    cudaStream_t stream
) {
    const int block_size = 256;
    int64_t grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 2147483647) {
        // Handle very large grids by iterative grid-stride loop; but here total_elements limited.
        // For safety, we can use a grid-stride loop in kernel and set grid to min(needed, max).
        // However, Simpler: use 1D grid with <= 2^31-1; assume not exceeded.
    }
    masked_fill_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        output, input, total_elements, mask_value, B, C, H, W, stride_b, stride_c, stride_h, stride_w
    );
}

// Host launcher for custom_cat along dim 1
// output: contiguous float tensor with shape [B, C1+C2, H, W]
// x, y: input tensors with their shapes/strides. C1 = x channel dim, C2 = y channel dim (both same H,W,B)
void launch_cat_dim1(
    float* output,
    const float* x,
    const float* y,
    int64_t B, int64_t C1, int64_t C2, int64_t H, int64_t W,
    int64_t x_stride_b, int64_t x_stride_c, int64_t x_stride_h, int64_t x_stride_w,
    int64_t y_stride_b, int64_t y_stride_c, int64_t y_stride_h, int64_t y_stride_w,
    cudaStream_t stream
) {
    int64_t total_output_elements = B * (C1 + C2) * H * W;
    const int block_size = 256;
    int64_t grid_size = (total_output_elements + block_size - 1) / block_size;
    cat_dim1_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        output, x, y, total_output_elements, B, C1, C2, H, W,
        x_stride_b, x_stride_c, x_stride_h, x_stride_w,
        y_stride_b, y_stride_c, y_stride_h, y_stride_w
    );
}

// Export C functions for binding layer
// Use extern "C" to avoid name mangling
extern "C" {

void custom_masked_fill_launcher(
    float* output,
    const float* input,
    int64_t total_elements,
    float mask_value,
    int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t stride_b, int64_t stride_c, int64_t stride_h, int64_t stride_w,
    cudaStream_t stream
) {
    launch_masked_fill(output, input, total_elements, mask_value, B, C, H, W,
                       stride_b, stride_c, stride_h, stride_w, stream);
}

void custom_cat_launcher(
    float* output,
    const float* x,
    const float* y,
    int64_t B, int64_t C1, int64_t C2, int64_t H, int64_t W,
    int64_t x_stride_b, int64_t x_stride_c, int64_t x_stride_h, int64_t x_stride_w,
    int64_t y_stride_b, int64_t y_stride_c, int64_t y_stride_h, int64_t y_stride_w,
    cudaStream_t stream
) {
    launch_cat_dim1(output, x, y, B, C1, C2, H, W,
                    x_stride_b, x_stride_c, x_stride_h, x_stride_w,
                    y_stride_b, y_stride_c, y_stride_h, y_stride_w, stream);
}

} // extern "C"
