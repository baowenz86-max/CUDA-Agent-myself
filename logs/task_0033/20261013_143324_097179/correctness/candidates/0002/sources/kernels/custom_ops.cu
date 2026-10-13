// CUDA kernels for conv_transpose1d, layer_norm and addcmul operations
#include <cuda_runtime.h>

// ConvTranspose1d kernel: computes output = transposed_conv(input, weight) + bias
// Input x shape: [batch, in_channels, length]
// weight shape: [in_channels, out_channels, kernel_size]
// bias shape: [out_channels] (optional, may be null)
// Output shape: [batch, out_channels, L_out] where L_out = (length-1)*stride - 2*padding + kernel_size + output_padding
// We use a naive direct convolution approach:
// For each output element (b, oc, ol):
//   sum over ic and k where: input_index = (ol + padding - k) / stride is integer and within [0, length-1]
//   Specifically, the relation: input_pos = (ol + padding - k) / stride, and must be integer and  0 <= input_pos < length.
//   Then output[b, oc, ol] = sum_{ic, k} input[b, ic, input_pos] * weight[ic, oc, k]
// After computing sum, add bias[oc] if bias is provided.

__global__ void conv_transpose1d_kernel(const float* __restrict__ input,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias,
                                        float* __restrict__ output,
                                        int batch,
                                        int in_channels,
                                        int out_channels,
                                        int length,
                                        int L_out,
                                        int kernel_size,
                                        int stride,
                                        int padding,
                                        int output_padding) { // output_padding not used in computation, output already sized with it; extra positions stay zero if initialized to zero, or we handle explicitly.
    int ol = blockIdx.x * blockDim.x + threadIdx.x;
    if (ol >= L_out) return;

    int b = blockIdx.y;
    int oc = blockIdx.z;

    if (b >= batch || oc >= out_channels) return;

    float sum = 0.0f;
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int k = 0; k < kernel_size; ++k) {
            int numerator = ol + padding - k;
            if (numerator % stride != 0) continue;
            int il = numerator / stride;
            if (il >= 0 && il < length) {
                int input_idx = (b * in_channels + ic) * length + il;
                int weight_idx = (ic * out_channels + oc) * kernel_size + k;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    if (bias != nullptr) {
        sum += bias[oc];
    }
    int output_idx = (b * out_channels + oc) * L_out + ol;
    output[output_idx] = sum;
}

// LayerNorm: normalizes over the last two dimensions (out_channels, length) per batch.
// Input x shape: [batch, out_channels, length], contiguous in memory.
// Weight and bias shape: [out_channels, length], contiguous in memory.
// eps is used for numerical stability.
// The mean and variance are computed per batch row of size = out_channels * length.
// Fused kernel: one block per row, block_size=256, grid-stride loop over row elements.
__global__ void layer_norm_kernel(const float* __restrict__ x,
                                  const float* __restrict__ weight,
                                  const float* __restrict__ bias,
                                  float* __restrict__ y,
                                  int row_size,
                                  int num_rows,
                                  float eps) {
    extern __shared__ float sdata[]; // size = 2 * blockDim.x
    float* s_sum = sdata;
    float* s_sq_sum = sdata + blockDim.x;

    int row = blockIdx.x;
    if (row >= num_rows) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    s_sum[tid] = 0.0f;
    s_sq_sum[tid] = 0.0f;

    const float* x_row = x + (size_t)row * row_size;

    // Step 1: accumulate sum and sum of squares over row
    for (int idx = tid; idx < row_size; idx += block_size) {
        float val = x_row[idx];
        s_sum[tid] += val;
        s_sq_sum[tid] += val * val;
    }

    __syncthreads();

    // Reduce to thread 0
    for (int offset = block_size / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
            s_sq_sum[tid] += s_sq_sum[tid + offset];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / row_size;
    float var = s_sq_sum[0] / row_size - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Step 2: normalize and apply weight/bias
    // weight and bias are 1D arrays of length row_size (same for all rows)
    for (int idx = tid; idx < row_size; idx += block_size) {
        float val = x_row[idx];
        float norm_val = (val - mean) * inv_std;
        y[row * (size_t)row_size + idx] = norm_val * weight[idx] + bias[idx];
    }
}

// Addcmul: output = input + value * tensor1 * tensor2
__global__ void addcmul_kernel(const float* __restrict__ input,
                               const float* __restrict__ tensor1,
                               const float* __restrict__ tensor2,
                               float* __restrict__ output,
                               int total_elements,
                               float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        output[idx] = input[idx] + value * tensor1[idx] * tensor2[idx];
    }
}

// Launcher functions (extern "C" for binding)
extern "C" void conv_transpose_launcher(const float* input,
                                        const float* weight,
                                        const float* bias,
                                        float* output,
                                        int batch,
                                        int in_channels,
                                        int out_channels,
                                        int length,
                                        int L_out,
                                        int kernel_size,
                                        int stride,
                                        int padding,
                                        int output_padding,
                                        cudaStream_t stream) {
    dim3 block(256);
    dim3 grid((L_out + block.x - 1) / block.x, batch, out_channels);
    conv_transpose1d_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output, batch, in_channels, out_channels, length, L_out,
        kernel_size, stride, padding, output_padding);
}

extern "C" void layer_norm_launcher(const float* x,
                                    const float* weight,
                                    const float* bias,
                                    float* y,
                                    int batch,
                                    int out_channels,
                                    int length,
                                    float eps,
                                    cudaStream_t stream) {
    int row_size = out_channels * length;
    int num_rows = batch;
    int block_size = 256;
    dim3 grid(num_rows);
    size_t shared_mem = 2 * block_size * sizeof(float);
    layer_norm_kernel<<<grid, block_size, shared_mem, stream>>>(
        x, weight, bias, y, row_size, num_rows, eps);
}

extern "C" void addcmul_launcher(const float* input,
                                 const float* tensor1,
                                 const float* tensor2,
                                 float* output,
                                 int total_elements,
                                 float value,
                                 cudaStream_t stream) {
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    addcmul_kernel<<<grid_size, block_size, 0, stream>>>(
        input, tensor1, tensor2, output, total_elements, value);
}
