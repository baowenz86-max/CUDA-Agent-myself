#include <cuda_runtime.h>
#include <cfloat>

// Softmax kernel (dim=1, 3D input)
__global__ void softmax_kernel_dim1_3d(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int L,
    long long input_stride_b, long long input_stride_c, long long input_stride_l
) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (b >= B || l >= L) return;

    const float* input_base = input + b * input_stride_b + l * input_stride_l;
    float max_val = -FLT_MAX;
    for (int c = 0; c < C; ++c) {
        float val = input_base[c * input_stride_c];
        if (val > max_val) max_val = val;
    }
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        sum += expf(input_base[c * input_stride_c] - max_val);
    }
    float inv_sum = 1.0f / sum;
    float* output_base = output + b * (C * L) + l;
    for (int c = 0; c < C; ++c) {
        float val = expf(input_base[c * input_stride_c] - max_val) * inv_sum;
        output_base[c * L] = val;
    }
}

// ConvTranspose1d kernel: direct computation with gather from input and weight, plus bias.
// Input: [B, C_in, L_in], weight: [C_in, C_out, K], bias: [C_out]
// Output: [B, C_out, L_out] where L_out = (L_in-1)*stride - 2*padding + K + output_padding
// Each thread computes one output element (b, oc, ol).
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int B, int C_in, int C_out, int L_in, int L_out, int K,
    int stride, int padding, int output_padding
) {
    int ol = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z * blockDim.z + threadIdx.z;
    if (b >= B || oc >= C_out || ol >= L_out) return;

    float acc = 0.0f;
    // For each input channel and kernel position that contributes to this output position.
    for (int ic = 0; ic < C_in; ++ic) {
        for (int k = 0; k < K; ++k) {
            // Formula: for transposed conv, input index = (ol + padding - k) / stride, if divisible and in range.
            int numerator = ol + padding - k;
            if (numerator < 0) continue;
            if (numerator % stride != 0) continue;
            int il = numerator / stride;
            if (il >= L_in) continue;
            // weight layout: [C_in, C_out, K] -> index ic*C_out*K + oc*K + k
            float w = weight[ic * C_out * K + oc * K + k];
            float x = input[b * C_in * L_in + ic * L_in + il];
            acc += w * x;
        }
    }
    // Add bias
    acc += bias[oc];
    output[b * C_out * L_out + oc * L_out + ol] = acc;
}

// Launcher for softmax
extern "C" void softmax_forward_float32(
    const float* input,
    float* output,
    int B, int C, int L,
    long long input_stride_b,
    long long input_stride_c,
    long long input_stride_l,
    cudaStream_t stream
) {
    dim3 block(32, 4);
    int grid_x = (L + block.x - 1) / block.x;
    int grid_y = (B + block.y - 1) / block.y;
    dim3 grid(grid_x, grid_y);
    softmax_kernel_dim1_3d<<<grid, block, 0, stream>>>(
        input, output, B, C, L, input_stride_b, input_stride_c, input_stride_l
    );
}

// Launcher for conv_transpose1d
extern "C" void conv_transpose1d_forward_float32(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int B, int C_in, int C_out, int L_in, int L_out, int K,
    int stride, int padding, int output_padding,
    cudaStream_t stream
) {
    // Grid configuration: we use a 3D grid. For example, block (16, 8, 4).
    dim3 block(16, 8, 4);
    int grid_x = (L_out + block.x - 1) / block.x;
    int grid_y = (C_out + block.y - 1) / block.y;
    int grid_z = (B + block.z - 1) / block.z;
    dim3 grid(grid_x, grid_y, grid_z);
    conv_transpose1d_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output,
        B, C_in, C_out, L_in, L_out, K,
        stride, padding, output_padding
    );
}
