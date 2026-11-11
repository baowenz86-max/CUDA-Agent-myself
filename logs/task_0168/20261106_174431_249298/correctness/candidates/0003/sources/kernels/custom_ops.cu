#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA kernels and launchers for model_new.py
// All launchers use raw pointers and cudaStream_t

// -----------------------------------------------------------------------------
// add_bias_op: out[b,c,l] = in[b,c,l] + bias[c]
// bias shape [C,1] broadcast along L dimension
// -----------------------------------------------------------------------------
__global__ void add_bias_kernel(float* __restrict__ output,
                                const float* __restrict__ input,
                                const float* __restrict__ bias,
                                int batch_size, int channels, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * length;
    if (idx >= total) return;

    int l = idx % length;
    int c = (idx / length) % channels;
    output[idx] = input[idx] + bias[c];
}

extern "C" void add_bias_forward_launcher(float* output,
                                          const float* input,
                                          const float* bias,
                                          int batch_size, int channels, int length,
                                          cudaStream_t stream) {
    int total = batch_size * channels * length;
    if (total <= 0) return;
    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    add_bias_kernel<<<grid_size, block_size, 0, stream>>>(output, input, bias,
                                                          batch_size, channels, length);
}

// -----------------------------------------------------------------------------
// softmax_channel_op: softmax over channel dimension dim=1 for each (b,l)
// input shape [B,C,L], output same shape
// Two-pass softmax for numerical stability
// -----------------------------------------------------------------------------
__global__ void softmax_channel_kernel(float* __restrict__ output,
                                       const float* __restrict__ input,
                                       int batch_size, int channels, int length) {
    int b = blockIdx.y;
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch_size || l >= length) return;

    int base = b * channels * length + l;
    float max_val = -1e30f;
    for (int c = 0; c < channels; ++c) {
        float v = input[base + c * length];
        if (v > max_val) max_val = v;
    }
    float sum = 0.0f;
    for (int c = 0; c < channels; ++c) {
        sum += __expf(input[base + c * length] - max_val);
    }
    for (int c = 0; c < channels; ++c) {
        float val = __expf(input[base + c * length] - max_val) / sum;
        output[base + c * length] = val;
    }
}

extern "C" void softmax_channel_forward_launcher(float* output,
                                                 const float* input,
                                                 int batch_size, int channels, int length,
                                                 cudaStream_t stream) {
    if (batch_size <= 0 || channels <= 0 || length <= 0) return;
    const int block_size_x = 256;
    int grid_x = (length + block_size_x - 1) / block_size_x;
    int grid_y = batch_size;
    dim3 block(block_size_x);
    dim3 grid(grid_x, grid_y);
    softmax_channel_kernel<<<grid, block, 0, stream>>>(output, input,
                                                       batch_size, channels, length);
}

// -----------------------------------------------------------------------------
// scale_op: out = input * scalar
// -----------------------------------------------------------------------------
__global__ void scale_kernel(float* __restrict__ output,
                             const float* __restrict__ input,
                             float scalar, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        output[idx] = input[idx] * scalar;
    }
}

extern "C" void scale_forward_launcher(float* output,
                                      const float* input,
                                      float scalar,
                                      int total,
                                      cudaStream_t stream) {
    if (total <= 0) return;
    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    scale_kernel<<<grid_size, block_size, 0, stream>>>(output, input, scalar, total);
}

// -----------------------------------------------------------------------------
// conv_transpose1d: naive transposed convolution (no cuDNN to avoid dependency)
// Input: [B, C_in, L_in], weight: [C_in, C_out, K], stride, padding, output_padding
// Output: [B, C_out, L_out]
// L_out = (L_in - 1) * stride - 2 * padding + K + output_padding
// For each output position and output channel, sum over input channels and kernel positions where
//   input_index = (out_pos + padding - k) / stride, and (out_pos + padding - k) % stride == 0,
//   and 0 <= input_index < L_in.
// Implementation: each thread computes one output element (b, oc, opos)
// Use grid stride over total elements for simplicity.
// -----------------------------------------------------------------------------
__global__ void conv_transpose1d_kernel(float* __restrict__ output,
                                        const float* __restrict__ input,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias,  // may be nullptr
                                        int B, int C_in, int L_in,
                                        int C_out, int K,
                                        int stride, int padding, int output_padding,
                                        int L_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * C_out * L_out;
    if (idx >= total) return;

    int opos = idx % L_out;
    int oc = (idx / L_out) % C_out;
    int b = idx / (L_out * C_out);

    float acc = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int k = 0; k < K; ++k) {
            int pos = opos + padding - k;
            if (pos < 0) continue;
            if (pos % stride != 0) continue;
            int in_pos = pos / stride;
            if (in_pos >= L_in) continue;
            acc += input[b * C_in * L_in + c_in * L_in + in_pos] *
                   weight[c_in * C_out * K + oc * K + k];
        }
    }
    if (bias != nullptr) {
        acc += bias[oc];
    }
    output[idx] = acc;
}

extern "C" void conv_transpose_forward_launcher(float* output,
                                                const float* input,
                                                const float* weight,
                                                const float* bias,  // may be nullptr
                                                int B, int C_in, int L_in,
                                                int C_out, int K,
                                                int stride, int padding, int output_padding,
                                                cudaStream_t stream) {
    int L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    if (L_out <= 0) return;
    int total = B * C_out * L_out;
    if (total <= 0) return;
    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    conv_transpose1d_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input, weight, bias,
        B, C_in, L_in, C_out, K,
        stride, padding, output_padding, L_out);
}
