#include <cuda_runtime.h>

// CUDA kernels and launchers for ModelNew custom ops.
// Contract: only float32 tensors on CUDA; output bool for isin, float32 for maxpool.
// No PyTorch/ATen headers. All tensors passed as raw pointers plus sizes/strides.

// isin kernel: elementwise check if x value belongs to values (1D contiguous).
__global__ void isin_kernel(
    const float* __restrict__ x,
    const float* __restrict__ values,
    bool* __restrict__ out,
    long long total_elements,
    int values_size)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += stride) {
        float val = x[idx];
        bool found = false;
        for (int v = 0; v < values_size; ++v) {
            if (val == values[v]) {
                found = true;
                break;
            }
        }
        out[idx] = found;
    }
}

// isin launcher: accepts raw pointers and strides (x is non-contiguous possibly).
// x_strides and x_shape describe x; out_strides and out_shape must match x's shape
// but output is allocated contiguous? Contract says output preserves x's stride layout,
// so binding will allocate a tensor with x's strides. We'll compute linear index using
// high-D strides. To simplify, we flatten using total_elements and x's strides.
extern "C" void isin_custom_launcher(
    const float* x,
    const float* values,
    bool* out,
    const long long* x_shape,
    const long long* x_strides,
    int dim,
    int values_size,
    cudaStream_t stream)
{
    // dim is number of dimensions (3 for [16,128,16384])
    long long total_elements = 1;
    for (int i = 0; i < dim; ++i) total_elements *= x_shape[i];

    // Use linear index with x's strides to access each element. However, for simplicity,
    // we can assume x is contiguous? Contract says "supports strides" and output has same
    // stride layout as x. So x may be non-contiguous. We need to iterate over all logical
    // elements (flatten row-major) and use strides to map to actual memory.
    // We'll compute linear memory offset from multi-index using x_strides.
    // But kernel currently uses flat idx directly into x, which assumes contiguous.
    // Need to fix kernel to handle arbitrary strides. We'll modify kernel signature:
    // provide x_strides array; compute offset from logical index.
    // To keep launcher simple and avoid passing many arrays, we can compute offset inside
    // kernel using division/modulo, but that is slow. Alternatively, since dim is small (3),
    // we can unroll loops with compile-time known dim? But dim is runtime.
    // Better: pass x_strides as a single array to kernel; kernel calculates offset using
    // divisions. For performance, this may be okay for 33M elements.
    // We'll update kernel below to accept strides.
    // Actually, we can just use a stride array of max 8 dimensions, and loop over dims.
    // Let's rewrite kernel to handle arbitrary strides.
    // However, to avoid complexity, we note that x is likely contiguous in all tests,
    // but not guaranteed. We'll implement generic stride handling.
    
    // We'll implement a separate kernel that takes dim, shape, strides.
    // For now, the current isin_kernel defined above is not used; we will define a better one.
    // We'll replace it with a more general kernel after this comment.
    // --- new implementation ---
}

// Better approach: define a templated kernel for dim=3 since shape is fixed [16,128,16384]
// but we need flexibility. Use a loop over total_elements and compute multi-index.
__global__ void isin_kernel_strided(
    const float* __restrict__ x,
    const float* __restrict__ values,
    bool* __restrict__ out,
    long long total_elements,
    int values_size,
    long long dim0_stride, long long dim1_stride, long long dim2_stride)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long grid_stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += grid_stride) {
        // Assume dims: dim0, dim1, dim2, but we only have 3 for our use case.
        // We'll hardcode for 3D? Not flexible. Better use a generic method:
        // We'll compute offset using divisions by shapes, but we don't have shapes here.
        // Instead, we can pass shape and strides as arrays.
    }
}

// Generic strided isin kernel: pass arrays for shape and strides.
__global__ void isin_kernel_generic(
    const float* __restrict__ x,
    const float* __restrict__ values,
    bool* __restrict__ out,
    long long total_elements,
    int values_size,
    const long long* __restrict__ x_shape,
    const long long* __restrict__ x_strides,
    int dim)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long grid_stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += grid_stride) {
        // Compute multi-index from linear idx (row-major order)
        long long offset = 0;
        long long remaining = idx;
        for (int d = dim - 1; d >= 0; --d) {
            long long coord = remaining % x_shape[d];
            remaining /= x_shape[d];
            offset += coord * x_strides[d];
        }
        float val = x[offset];
        bool found = false;
        for (int v = 0; v < values_size; ++v) {
            if (val == values[v]) {
                found = true;
                break;
            }
        }
        out[idx] = found;  // output is contiguous bool (assuming binding allocates contiguous with x's shape)
        // But contract says output preserves x's stride layout. If x is non-contiguous, output should have same strides.
        // However, for bool, binding may allocate a contiguous tensor of same shape?
        // Contract says output has same shape and stride as x. So if x is non-contiguous, output should also be non-contiguous with same strides.
        // Our current out uses idx directly, which assumes contiguous. To be safe, we should also use out_strides.
        // We'll need to pass out_strides as well.
    }
}

// The above kernel still assumes output contiguous. We need out_strides as well.
// We'll define a final kernel that takes both x and out shape/strides, but to reduce complexity,
// we can assume that output is contiguous? The contract explicitly states: "输出与 x 相同形状和 stride（如果 x 非连续，输出也使用相同 stride 布局）".
// So we must preserve stride. We'll pass out_strides and compute offset for out as well.

// Final kernel for isin (handles arbitrary strides for both x and out, but out shape matches x shape).
__global__ void isin_kernel_strided_both(
    const float* __restrict__ x,
    const float* __restrict__ values,
    bool* __restrict__ out,
    long long total_elements,
    int values_size,
    const long long* __restrict__ x_shape,
    const long long* __restrict__ x_strides,
    const long long* __restrict__ out_strides,
    int dim)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long grid_stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += grid_stride) {
        long long remaining = idx;
        long long x_offset = 0;
        long long out_offset = 0;
        for (int d = dim - 1; d >= 0; --d) {
            long long coord = remaining % x_shape[d];
            remaining /= x_shape[d];
            x_offset += coord * x_strides[d];
            out_offset += coord * out_strides[d];
        }
        float val = x[x_offset];
        bool found = false;
        for (int v = 0; v < values_size; ++v) {
            if (val == values[v]) {
                found = true;
                break;
            }
        }
        out[out_offset] = found;
    }
}

// MaxPool1d kernel: operates on last dimension, for each output position, find max in window.
// Input x: shape [batch, channel, length], can be non-contiguous.
// Output y: shape [batch, channel, out_length], contiguous (contract says output is contiguous).
// We'll launch one thread per output element.
__global__ void maxpool1d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    long long x_batch_stride,
    long long x_channel_stride,
    long long x_length_stride)  // Usually 1, but may be different? Since x may be non-contiguous, strides may be arbitrary.
{
    // Compute output index: out_b, out_c, out_i
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long grid_stride = (long long)gridDim.x * blockDim.x;
    long long total_outputs = (long long)batch_size * channels * output_length;
    for (; idx < total_outputs; idx += grid_stride) {
        int out_i = idx % output_length;
        long long tmp = idx / output_length;
        int out_c = tmp % channels;
        int out_b = tmp / channels;

        // Input start index: out_i * stride - padding
        int start = out_i * stride - padding;
        // We need to consider dilation: window positions are start + k*dilation for k=0..kernel_size-1
        // For max, we iterate over kernel positions.
        float max_val = -3.402823466e+38F; // -FLT_MAX
        bool valid = false;
        for (int k = 0; k < kernel_size; ++k) {
            int in_pos = start + k * dilation;
            if (in_pos >= 0 && in_pos < input_length) {
                // Compute input memory offset
                long long in_offset = (long long)out_b * x_batch_stride + (long long)out_c * x_channel_stride + (long long)in_pos * x_length_stride;
                float val = x[in_offset];
                if (!valid || val > max_val) {
                    max_val = val;
                    valid = true;
                }
            }
        }
        // If no valid elements (e.g., entire window out of bounds), should not happen because ceil_mode false and padding 0, output_length ensures at least one valid? Actually with padding=0 and dilation=1, if out_i*stride >= input_length, no valid. But output_length calculation ensures out_i < output_length = floor((L - kernel_size)/stride)+1, so start < L - kernel_size +1? Let's verify: output_length = floor((16384-2)/3)+1 = 5461. Max out_i = 5460; start = 16380; window covers 16380,16381,16382 all within 0..16383. So valid always true. But to be safe, keep valid flag.
        y[idx] = max_val; // output is contiguous
    }
}

// Launcher for isin_custom
extern "C" void isin_custom_launcher(
    const float* x,
    const float* values,
    bool* out,
    const long long* x_shape,
    const long long* x_strides,
    const long long* out_strides,
    int dim,
    int values_size,
    long long total_elements,
    cudaStream_t stream)
{
    int threads = 256;
    long long blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid limit, but we use grid-stride loop so okay; actually we can set bigger? max grid x dimension for compute capability 8.0 is 2^31-1, but to be safe set to a reasonable number.
    blocks = min(blocks, (long long)65535); // use grid-stride loop to cover all
    
    isin_kernel_strided_both<<<(int)blocks, threads, 0, stream>>>(
        x, values, out, total_elements, values_size,
        x_shape, x_strides, out_strides, dim);
}

// Launcher for maxpool1d_custom
extern "C" void maxpool1d_custom_launcher(
    const float* x,
    float* y,
    int batch_size,
    int channels,
    int input_length,
    int output_length,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    long long x_batch_stride,
    long long x_channel_stride,
    long long x_length_stride,
    cudaStream_t stream)
{
    long long total_outputs = (long long)batch_size * channels * output_length;
    int threads = 256;
    long long blocks = (total_outputs + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    maxpool1d_kernel<<<(int)blocks, threads, 0, stream>>>(
        x, y, batch_size, channels, input_length, output_length,
        kernel_size, stride, padding, dilation,
        x_batch_stride, x_channel_stride, x_length_stride);
}
