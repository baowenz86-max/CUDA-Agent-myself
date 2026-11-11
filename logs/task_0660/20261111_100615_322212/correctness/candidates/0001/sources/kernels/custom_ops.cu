// kernels/custom_ops.cu
// CUDA kernels and launchers for constant_pad1d_forward and sum_dim2_forward
// No torch/ATen/pybind dependencies. Only raw pointers and cudaStream_t.

#include <cuda_runtime.h>

// Kernel for constant padding along the last dimension.
// Input: x with shape (batch * seq, width_in) where width_in = width
// Output: out with shape (batch * seq, width_out) where width_out = width + pad_left + pad_right
// Padded values are set to 'value' everywhere, then copy original input to correct offset.
// We assume input is contiguous, i.e., row-major with stride = width_in.
__global__ void constant_pad1d_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int total_rows,       // batch * seq
    int width_in,
    int width_out,
    int pad_left,
    int pad_right,
    float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total_elements = total_rows * width_out;

    for (int i = idx; i < total_elements; i += stride) {
        int col = i % width_out;
        int row = i / width_out;
        // Determine source column: original col = col - pad_left, valid if in [0, width_in-1]
        int src_col = col - pad_left;
        if (src_col >= 0 && src_col < width_in) {
            out[i] = in[row * width_in + src_col];
        } else {
            out[i] = value;
        }
    }
}

// Kernel for sum along last dimension (dim=2) for contiguous input of shape (batch, seq, width).
// Output shape (batch, seq). Each thread computes one output element, using grid-stride loop over rows.
__global__ void sum_dim2_kernel(
    float* __restrict__ out,
    const float* __restrict__ in,
    int total_rows,       // batch * seq
    int width)            // length of last dimension
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int row = idx; row < total_rows; row += stride) {
        const float* row_ptr = in + row * width;
        float sum = 0.0f;
        // Sequential sum for each row
        #pragma unroll 8
        for (int col = 0; col < width; ++col) {
            sum += row_ptr[col];
        }
        out[row] = sum;
    }
}

// Host launcher for constant padding.
// Parameters match contract: (x, pad_left, pad_right, value)
// 'stream' is provided by binding layer (cudaStream_t).
extern "C" void constant_pad1d_launcher(
    float* output,
    const float* input,
    int64_t batch,
    int64_t seq,
    int64_t width,
    int64_t pad_left,
    int64_t pad_right,
    double value,
    cudaStream_t stream)
{
    int64_t total_rows_64 = batch * seq;
    // Guard: total_rows fits in int; for spec, total_rows = 128*512 = 65536, safe.
    int total_rows = static_cast<int>(total_rows_64);
    int width_in = static_cast<int>(width);
    int width_out = static_cast<int>(width + pad_left + pad_right);
    int pad_left_int = static_cast<int>(pad_left);
    int pad_right_int = static_cast<int>(pad_right);
    float fill_value = static_cast<float>(value);

    // Number of elements to process in output
    int64_t numel = total_rows_64 * (width + pad_left + pad_right);

    // Configure grid and block, ensuring covering all elements
    const int block_size = 256;
    int64_t blocks_64 = (numel + block_size - 1) / block_size;
    // Cap grid dimension to avoid exceeding GPU limits (we use grid-stride loop, so can cap)
    int grid_size = static_cast<int>(blocks_64 > 65535 ? 65535 : blocks_64);
    if (grid_size < 1) grid_size = 1;

    constant_pad1d_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, total_rows, width_in, width_out, pad_left_int, pad_right_int, fill_value);
}

// Host launcher for sum along last dimension, output shape (batch, seq).
extern "C" void sum_dim2_launcher(
    float* output,
    const float* input,
    int64_t batch,
    int64_t seq,
    int64_t width,
    cudaStream_t stream)
{
    int64_t total_rows_64 = batch * seq;
    int total_rows = static_cast<int>(total_rows_64);
    int width_int = static_cast<int>(width);

    // Number of rows to process
    int64_t rows = total_rows_64;

    const int block_size = 256;
    int64_t blocks_64 = (rows + block_size - 1) / block_size;
    int grid_size = static_cast<int>(blocks_64 > 65535 ? 65535 : blocks_64);
    if (grid_size < 1) grid_size = 1;

    sum_dim2_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, total_rows, width_int);
}
