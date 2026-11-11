// custom_ops.cu
// CUDA kernels for max_reduce and layer_norm_forward
#include <cuda_runtime.h>
#include <stdint.h>

// Compute maximum along dim and its index
// Input: x (N, C, H, W) with strides sx0, sx1, sx2, sx3
// Output: values (N, 1, H, W), indices (N, 1, H, W) int64
// dim is always 1, keepdim always true, but verify in launcher
__global__ void max_reduce_kernel(
    const float* __restrict__ x,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int N, int C, int H, int W,
    int64_t sx0, int64_t sx1, int64_t sx2, int64_t sx3,
    int64_t sv0, int64_t sv1, int64_t sv2, int64_t sv3,
    int64_t si0, int64_t si1, int64_t si2, int64_t si3)
{
    int64_t total_outputs = (int64_t)N * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_stride = (int64_t)gridDim.x * blockDim.x;
    
    for (int64_t out_idx = idx; out_idx < total_outputs; out_idx += grid_stride) {
        int n = out_idx / ((int64_t)H * W);
        int rem = out_idx % ((int64_t)H * W);
        int h = rem / W;
        int w = rem % W;
        
        const float* base_ptr = x + n * sx0 + h * sx2 + w * sx3;
        float max_val = base_ptr[0];
        int max_idx = 0;
        for (int c = 1; c < C; ++c) {
            float val = base_ptr[c * sx1];
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
        }
        
        float* val_out = values + n * sv0 + h * sv2 + w * sv3;
        int64_t* idx_out = indices + n * si0 + h * si2 + w * si3;
        *val_out = max_val;
        *idx_out = (int64_t)max_idx;
    }
}

// LayerNorm over last dimension (size D = W) per (N,H) position
// Input: x (N, 1, H, D) with strides, weight (D), bias (D)
// Output: y same shape as input (strides same as input)
__global__ void layer_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float eps,
    int N, int H, int D,
    int64_t sx0, int64_t sx1, int64_t sx2, int64_t sx3,
    int64_t sy0, int64_t sy1, int64_t sy2, int64_t sy3)
{
    int64_t total_positions = (int64_t)N * H;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_stride = (int64_t)gridDim.x * blockDim.x;
    
    for (int64_t pos = idx; pos < total_positions; pos += grid_stride) {
        int n = pos / H;
        int h = pos % H;
        
        const float* row = input + n * sx0 + h * sx2; // row length D, stride sx3 between elements
        // Compute mean and variance
        float sum = 0.0f;
        float sum_sq = 0.0f;
        for (int d = 0; d < D; ++d) {
            float val = row[d * sx3];
            sum += val;
            sum_sq += val * val;
        }
        float mean = sum / D;
        float var = sum_sq / D - mean * mean;
        float inv_std = rsqrtf(var + eps);
        
        float* out_row = output + n * sy0 + h * sy2;
        for (int d = 0; d < D; ++d) {
            float val = row[d * sx3];
            float normalized = (val - mean) * inv_std;
            float result = normalized * weight[d] + bias[d];
            out_row[d * sy3] = result;
        }
    }
}

// Launcher for max_reduce
// x: input tensor data pointer
// N, C, H, W: dimensions
// strides: array of 4 int64 for input, values, indices
// values, indices: output pointers
// stream: CUDA stream
extern "C" void max_reduce_launcher(
    const float* x,
    float* values,
    int64_t* indices,
    int N, int C, int H, int W,
    const int64_t* x_strides,
    const int64_t* values_strides,
    const int64_t* indices_strides,
    cudaStream_t stream)
{
    int64_t total_outputs = (int64_t)N * H * W;
    int block_size = 256;
    int64_t grid_size = (total_outputs + block_size - 1) / block_size;
    // Cap grid to avoid excessive launches
    if (grid_size > 65535) grid_size = 65535;
    
    max_reduce_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        x, values, indices,
        N, C, H, W,
        x_strides[0], x_strides[1], x_strides[2], x_strides[3],
        values_strides[0], values_strides[1], values_strides[2], values_strides[3],
        indices_strides[0], indices_strides[1], indices_strides[2], indices_strides[3]);
}

// Launcher for layer_norm_forward
// input: input tensor data pointer
// weight, bias: parameter pointers (size D)
// output: output tensor data pointer (same shape as input)
// N, H, D: dimensions (input shape [N, 1, H, D])
// strides: input and output strides arrays
// eps: epsilon
extern "C" void layer_norm_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float eps,
    int N, int H, int D,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream)
{
    int64_t total_positions = (int64_t)N * H;
    int block_size = 256;
    int64_t grid_size = (total_positions + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    
    layer_norm_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        input, weight, bias, output, eps,
        N, H, D,
        input_strides[0], input_strides[1], input_strides[2], input_strides[3],
        output_strides[0], output_strides[1], output_strides[2], output_strides[3]);
}
