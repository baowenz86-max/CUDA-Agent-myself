#include <cuda_runtime.h>

// ============================================================================
// CUDA kernels for softmax (dim=1, channels_last), leaky_relu, add_bias,
// and direct convolution (channels_last input/output)
// ============================================================================

// ----------------------------------------------------------------------------
// Softmax along channel dimension (dim=1) for channels_last layout (NHWC)
// Input x: shape [N, C, H, W] but stored as NHWC, i.e., element (n,h,w,c)
// Channels are contiguous in memory with stride = 1
// Output: same shape and layout, softmax over C for each (n,h,w)
// ----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void softmax_channel_last_kernel_impl(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W)
{
    int total_spatial = N * H * W;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_spatial;
         idx += blockDim.x * gridDim.x)
    {
        // Compute n, h, w from idx
        int n = idx / (H * W);
        int rem = idx % (H * W);
        int h = rem / W;
        int w = rem % W;

        // Base pointer for this (n, h, w) location
        const float* in_ptr = input + ((n * H + h) * W + w) * C;
        float* out_ptr = output + ((n * H + h) * W + w) * C;

        // Find max over channels for numerical stability
        float max_val = -1e30f;
        for (int c = 0; c < C; ++c) {
            float val = in_ptr[c];
            if (val > max_val) max_val = val;
        }

        // Compute exp(x - max) and sum
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            float val = __expf(in_ptr[c] - max_val);
            out_ptr[c] = val;
            sum += val;
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < C; ++c) {
            out_ptr[c] *= inv_sum;
        }
    }
}

extern "C" void softmax_channel_last_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int total_spatial = N * H * W;
    const int block_size = 256;
    int grid_size = (total_spatial + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    softmax_channel_last_kernel_impl<block_size><<<grid_size, block_size, 0, stream>>>(
        output, input, N, C, H, W);
}

// ----------------------------------------------------------------------------
// LeakyReLU elementwise: y = x >= 0 ? x : negative_slope * x
// Works for any contiguous or non-contiguous tensor via element index
// The caller ensures output and input have same numel and dtype float32
// ----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void leaky_relu_kernel_impl(
    float* __restrict__ output,
    const float* __restrict__ input,
    int numel,
    float negative_slope)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        float val = input[idx];
        output[idx] = (val >= 0.0f) ? val : negative_slope * val;
    }
}

extern "C" void leaky_relu_launcher(
    float* output,
    const float* input,
    int numel,
    float negative_slope,
    cudaStream_t stream)
{
    const int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    leaky_relu_kernel_impl<block_size><<<grid_size, block_size, 0, stream>>>(
        output, input, numel, negative_slope);
}

// ----------------------------------------------------------------------------
// Add bias broadcast: output[n,c,h,w] = input[n,c,h,w] + bias[c]
// Input shape [N, C, H, W] with channels_last layout (NHWC)
// bias shape [C, 1, 1] but passed as contiguous array of length C
// We compute per element index mapping to (n,c,h,w) and add bias[c]
// ----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void add_bias_kernel_impl(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ bias,
    int N, int C, int H, int W)
{
    int numel = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        // Decode idx to (n,c,h,w) for NHWC layout
        int c = idx % C;
        output[idx] = input[idx] + bias[c];
    }
}

extern "C" void add_bias_launcher(
    float* output,
    const float* input,
    const float* bias,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int numel = N * C * H * W;
    const int block_size = 256;
    int grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    add_bias_kernel_impl<block_size><<<grid_size, block_size, 0, stream>>>(
        output, input, bias, N, C, H, W);
}

// ----------------------------------------------------------------------------
// Direct convolution 2D with channels_last input/output (NHWC)
// Weight is OIHW (C_out, C_in, kH, kW), contiguous.
// Output shape: [N, C_out, H_out, W_out] in NHWC layout.
// Each thread computes one output element (n, h_out, w_out, c_out).
// We use global memory for input and weight. Since weight is small, we can
// use constant memory or shared memory, but for simplicity we read directly.
// ----------------------------------------------------------------------------
template <int BLOCK_SIZE>
__global__ void conv2d_direct_kernel_impl(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C_in, int C_out, int H_in, int W_in,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w)
{
    int total_output = N * C_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_output) return;

    // Decode idx into (n, c_out, h_out, w_out) for NHWC layout:
    // NHWC memory order: n major, then h, then w, then c minor.
    // So index = ((n * H_out + h_out) * W_out + w_out) * C_out + c_out
    int c_out = idx % C_out;
    int tmp = idx / C_out;
    int w_out = tmp % W_out;
    tmp /= W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int n = tmp;

    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_in = h_out * stride_h - pad_h + kh;
            if (h_in < 0 || h_in >= H_in) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int w_in = w_out * stride_w - pad_w + kw;
                if (w_in < 0 || w_in >= W_in) continue;
                // Input index in NHWC:
                int input_idx = ((n * H_in + h_in) * W_in + w_in) * C_in + c_in;
                // Weight index in OIHW layout: ((c_out * C_in + c_in) * kernel_h + kh) * kernel_w + kw
                int weight_idx = ((c_out * C_in + c_in) * kernel_h + kh) * kernel_w + kw;
                sum += input[input_idx] * weight[weight_idx];
            }
        }
    }
    // Add bias
    sum += bias[c_out];

    // Output index in NHWC:
    int output_idx = ((n * H_out + h_out) * W_out + w_out) * C_out + c_out;
    output[output_idx] = sum;
}

extern "C" void conv2d_direct_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C_in, int C_out, int H_in, int W_in,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    cudaStream_t stream)
{
    int total_output = N * C_out * H_out * W_out;
    const int block_size = 256;
    int grid_size = (total_output + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    conv2d_direct_kernel_impl<block_size><<<grid_size, block_size, 0, stream>>>(
        output, input, weight, bias,
        N, C_in, C_out, H_in, W_in, H_out, W_out,
        kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w);
}

// Ensure there is at least one symbol defined for the .cu compilation unit.
static __global__ void dummy_kernel() {}
