// kernels/custom_ops.cu
// Contains CUDA kernels and launchers for custom_conv_transpose1d and custom_histc.
// This file must not include any torch/ATen/pybind headers.
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// -----------------------------------------------------------------------------
// Histogram kernel: computes histogram of a float tensor with given bins, min, max.
// The histogram is computed by atomically incrementing counts per bin.
// Input: flat array of size total_elements.
// Output: hist array of size bins, initialized to zero.
// Kernel uses grid-stride loop to handle arbitrarily large inputs.
// -----------------------------------------------------------------------------
__global__ void histc_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             int total_elements,
                             int bins,
                             float min_val,
                             float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float bin_width = (max_val - min_val) / bins;

    for (int i = idx; i < total_elements; i += stride) {
        float val = input[i];
        if (val >= min_val && val <= max_val) {
            int bin = (int)((val - min_val) / bin_width);
            if (bin == bins) bin = bins - 1; // include max_val in last bin
            if (bin >= 0 && bin < bins) {
                atomicAdd(&output[bin], 1.0f);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Host launcher for custom_histc.
// Parameters:
//   input: pointer to input tensor data (float32, contiguous)
//   output: pointer to output tensor data (float32, size bins)
//   total_elements: number of elements in input
//   bins: number of histogram bins
//   min_val, max_val: histogram range
//   stream: CUDA stream to execute on
// -----------------------------------------------------------------------------
extern "C" void custom_histc_launcher(const float* input,
                                      float* output,
                                      int total_elements,
                                      int bins,
                                      float min_val,
                                      float max_val,
                                      cudaStream_t stream) {
    // Initialize output to zero
    cudaMemsetAsync(output, 0, bins * sizeof(float), stream);

    // Launch configuration
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    // Cap grid size to avoid exceeding max grid dimension
    const int max_grid_size_x = 65535; // common limit, adjust if needed
    if (num_blocks > max_grid_size_x) {
        num_blocks = max_grid_size_x;
    }

    if (total_elements > 0) {
        histc_kernel<<<num_blocks, block_size, 0, stream>>>(
            input, output, total_elements, bins, min_val, max_val);
    }

    // Error check can be added by caller if needed, but we can check here:
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) { ... }
}

// -----------------------------------------------------------------------------
// ConvTranspose1d via custom CUDA kernel
// Implements transposed convolution for 1D data (N, C_in, L_in) -> (N, C_out, L_out).
// Supports groups=1, dilation=1. Handles stride, padding, output_padding.
// Parameters:
//   input:   [N, C_in, L_in]
//   weight:  [C_in, C_out, kernel_size]  (for groups=1)
//   bias:    [C_out] or nullptr
//   output:  [N, C_out, L_out]
//   N, C_in, C_out, L_in, L_out, kernel_size, stride, padding, output_padding, groups, dilation
//   stream: CUDA stream
// -----------------------------------------------------------------------------

__global__ void conv_transpose1d_kernel(const float* __restrict__ input,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias,
                                        float* __restrict__ output,
                                        int N, int C_in, int C_out,
                                        int L_in, int L_out,
                                        int kernel_size, int stride, int padding, int output_padding) {
    // Each thread computes one output element (n, oc, ol)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out_elements = N * C_out * L_out;
    if (idx >= total_out_elements) return;

    int ol = idx % L_out;
    int tmp = idx / L_out;
    int oc = tmp % C_out;
    int n = tmp / C_out;

    float sum = 0.0f;

    // Iterate over input channels and kernel positions that contribute to this output
    for (int ic = 0; ic < C_in; ++ic) {
        for (int ki = 0; ki < kernel_size; ++ki) {
            // Input position corresponding to output ol and kernel index ki:
            // ol = (il - 1) * stride - padding + ki * dilation + output_padding? Actually standard formula:
            // output[ol] = bias[oc] + sum_{ic, ki} input[ic, il] * weight[ic, oc, ki]
            // where ol = (il - 1) * stride - padding + ki * dilation + output_padding? Wait, output_padding only adds zeros at the end, not part of summation.
            // Let's derive: ConvTranspose1d output size: L_out = (L_in - 1) * stride - 2*padding + kernel_size + output_padding.
            // For each output position ol, contributions come from input positions il that satisfy:
            // ol = il * stride - padding + ki * dilation + output_padding?  Actually no output_padding in the summation.
            // The relation: ol = il * stride - padding + ki * dilation, where ki ranges 0..kernel_size-1.
            // However, output_padding simply extends the output size by adding extra positions at the end that would be computed as if there were extra input zeros? But all those extra positions are zero (except bias). So output_padding positions at the end receive only bias.
            // We'll handle output_padding by simply not computing for those extra positions; they'll be set to bias (or 0 if no bias).
            // In kernel, we only compute for ol < L_out - output_padding, and for the rest we set bias (if present) or 0.
            // But wait, the formula without output_padding gives L_out_without_op = (L_in - 1)*stride - 2*padding + kernel_size.
            // With output_padding, L_out = L_out_without_op + output_padding.
            // For ol >= L_out_without_op, there is no corresponding il, so output = bias[oc] (or 0).
            // So first compute the base output length: int L_out_base = L_out - output_padding;
            // For ol < L_out_base, we compute using the loop below, but careful: the loop condition il must be valid and the formula must be valid.
            // The formula: ol = il * stride - padding + ki, with ki from 0..kernel_size-1.
            // Solve for il: il = (ol + padding - ki) / stride.
            // il must be integer and 0 <= il < L_in.
            // So in the loop over ki and ic, we check if (ol + padding - ki) % stride == 0, then il = (ol + padding - ki) / stride, if il in range.
            // For simplicity, we loop over il as well? Or use the above condition.
            // We'll loop over ki and compute il accordingly.
            // Note: dilation=1, so dilation factor is 1.
            int numerator = ol + padding - ki;
            if (numerator >= 0 && numerator % stride == 0) {
                int il = numerator / stride;
                if (il >= 0 && il < L_in) {
                    // Valid input position
                    float input_val = input[((n * C_in + ic) * L_in) + il];
                    float weight_val = weight[((ic * C_out) + oc) * kernel_size + ki];
                    sum += input_val * weight_val;
                }
            }
        }
    }

    // Add bias if present
    if (bias != nullptr) {
        sum += bias[oc];
    }

    output[idx] = sum;
}

extern "C" void custom_conv_transpose1d_launcher(const float* input,
                                                const float* weight,
                                                const float* bias,
                                                float* output,
                                                int N,
                                                int C_in,
                                                int C_out,
                                                int L_in,
                                                int L_out,
                                                int kernel_size,
                                                int stride,
                                                int padding,
                                                int output_padding,
                                                int groups,
                                                int dilation,
                                                cudaStream_t stream) {
    // Only support groups=1, dilation=1 for now
    if (groups != 1 || dilation != 1) {
        fprintf(stderr, "custom_conv_transpose1d_launcher only supports groups=1, dilation=1\n");
        return;
    }

    int total_out_elements = N * C_out * L_out;
    int block_size = 256;
    int num_blocks = (total_out_elements + block_size - 1) / block_size;

    if (total_out_elements > 0) {
        conv_transpose1d_kernel<<<num_blocks, block_size, 0, stream>>>(
            input, weight, bias, output,
            N, C_in, C_out, L_in, L_out,
            kernel_size, stride, padding, output_padding);
    }

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "custom_conv_transpose1d_launcher: CUDA error after launch: %s\n", cudaGetErrorString(err));
    }
}
