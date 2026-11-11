#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel for 2D max pooling.
// Each thread computes one output element (or grids-stride loop for larger sizes).
// Input may have arbitrary strides, so we compute input address using the provided strides.
__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int64_t stride_n, int64_t stride_c, int64_t stride_h_in, int64_t stride_w_in,
    int64_t stride_n_out, int64_t stride_c_out, int64_t stride_h_out, int64_t stride_w_out)
{
    // Total output elements = N * C * H_out * W_out
    int64_t total_elements = (int64_t)N * C * H_out * W_out;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_size = (int64_t)gridDim.x * blockDim.x;

    for (int64_t linear_idx = idx; linear_idx < total_elements; linear_idx += grid_size) {
        // decompose linear_idx into (n, c, h_out, w_out)
        int w_out = linear_idx % W_out;
        int64_t tmp = linear_idx / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        // Compute input window start (top-left) coordinates
        int h_start = h_out * stride_h - padding_h;
        int w_start = w_out * stride_w - padding_w;

        float max_val = -1e38f; // -FLT_MAX
        bool found = false;

        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_in = h_start + kh * dilation_h;
            if (h_in < 0 || h_in >= H_in) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int w_in = w_start + kw * dilation_w;
                if (w_in < 0 || w_in >= W_in) continue;
                // compute input linear index using input strides
                int64_t input_idx = (int64_t)n * stride_n + (int64_t)c * stride_c
                                   + (int64_t)h_in * stride_h_in + (int64_t)w_in * stride_w_in;
                float val = input[input_idx];
                if (!found || val > max_val) {
                    max_val = val;
                    found = true;
                }
            }
        }

        // If no valid input (should not happen for valid pooling configs, but to be safe)
        // In PyTorch, output is undefined if all inputs are padding and no valid element; here we set to -inf.
        // However, for standard max_pool2d with ceil_mode=False, windows always have at least one valid element.
        if (!found) {
            max_val = -__int_as_float(0x7f800000); // -inf
        }

        // compute output linear index using output strides
        int64_t output_idx = (int64_t)n * stride_n_out + (int64_t)c * stride_c_out
                            + (int64_t)h_out * stride_h_out + (int64_t)w_out * stride_w_out;
        output[output_idx] = max_val;
    }
}

// Launcher function: no PyTorch dependencies, called from binding.cpp
extern "C" void maxpool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int kernel_size, int stride, int padding, int dilation, bool ceil_mode,
    int64_t stride_n, int64_t stride_c, int64_t stride_h_in, int64_t stride_w_in,
    int64_t stride_n_out, int64_t stride_c_out, int64_t stride_h_out, int64_t stride_w_out,
    cudaStream_t stream)
{
    // Compute output dimensions according to PyTorch formula:
    // H_out = floor((H_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
    // If ceil_mode=True, use ceil instead of floor.
    int H_out, W_out;
    if (ceil_mode) {
        H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) / stride + 1;
        W_out = (W_in + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) / stride + 1;
    } else {
        H_out = (H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        W_out = (W_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    }

    // Total number of output elements
    int64_t total_elements = (int64_t)N * C * H_out * W_out;

    // Choose block size and grid size
    int block_size = 256;
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    // Cap grid size to avoid exceeding max grid dimensions (2^31-1)
    const int max_grid_size = 65535; // common limit for x dimension on many GPUs
    if (grid_size > max_grid_size) {
        grid_size = max_grid_size;
    }

    // Launch kernel
    maxpool2d_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output,
        N, C, H_in, W_in,
        H_out, W_out,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
        dilation, dilation,
        stride_n, stride_c, stride_h_in, stride_w_in,
        stride_n_out, stride_c_out, stride_h_out, stride_w_out);
}
