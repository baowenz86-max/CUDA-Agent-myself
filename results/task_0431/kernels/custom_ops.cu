#include <cuda_runtime.h>
#include <cmath>

// Softplus forward kernel: y = log(1 + exp(x)), with threshold handling
__global__ void softplus_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        // PyTorch Softplus with beta=1, threshold=20: if x > 20, return x; else log1p(exp(x))
        output[i] = (x > 20.0f) ? x : log1pf(expf(x));
    }
}

// SiLU forward kernel: y = x * sigmoid(x)
__global__ void silu_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

// ReflectionPad2d kernel: applies reflection padding with given pad values (left, right, top, bottom)
// Input is assumed contiguous, shape: [N, C, H, W]
// Output shape: [N, C, H+top+bottom, W+left+right]
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t pad_left, int64_t pad_right, int64_t pad_top, int64_t pad_bottom,
    int64_t H_out, int64_t W_out, int64_t total_elements)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        // Decompose linear index into n, c, h_out, w_out (assuming contiguous output)
        int64_t tmp = linear;
        int64_t w_out = tmp % W_out;
        tmp /= W_out;
        int64_t h_out = tmp % H_out;
        tmp /= H_out;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // Compute source indices with reflection
        int64_t w_in = w_out - pad_left;
        if (w_in < 0) {
            w_in = -w_in;  // reflection for left padding: index -1 -> 1, -2 -> 2, etc.
        } else if (w_in >= W_in) {
            w_in = 2 * W_in - 2 - w_in;  // reflection for right padding
        }

        int64_t h_in = h_out - pad_top;
        if (h_in < 0) {
            h_in = -h_in;
        } else if (h_in >= H_in) {
            h_in = 2 * H_in - 2 - h_in;
        }

        // Clamp to valid range (reflection formula may yield edge cases when pad exceeds size)
        if (w_in < 0) w_in = 0;
        if (w_in >= W_in) w_in = W_in - 1;
        if (h_in < 0) h_in = 0;
        if (h_in >= H_in) h_in = H_in - 1;

        int64_t input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
        output[linear] = input[input_idx];
    }
}

// Launchers (extern "C" for binding layer)
extern "C" void softplus_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream)
{
    const int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    // Cap blocks to avoid grid limit and use grid-stride loop inside kernel
    if (blocks > 65535) blocks = 65535;
    softplus_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void silu_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream)
{
    const int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    silu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void reflection_pad2d_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H_in, int64_t W_in,
    int64_t pad_left, int64_t pad_right, int64_t pad_top, int64_t pad_bottom,
    int64_t H_out, int64_t W_out,
    cudaStream_t stream)
{
    int64_t total_elements = N * C * H_out * W_out;
    const int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H_in, W_in,
        pad_left, pad_right, pad_top, pad_bottom,
        H_out, W_out, total_elements);
}
