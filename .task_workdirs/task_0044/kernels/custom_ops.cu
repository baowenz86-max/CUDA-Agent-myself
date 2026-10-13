#include <cuda_runtime.h>

// Elementwise leaky ReLU kernel.
__global__ void leaky_relu_kernel(float* output, const float* input, float negative_slope, long long total_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < total_elements; i += stride) {
        float val = input[i];
        output[i] = (val > 0.0f) ? val : (val * negative_slope);
    }
}

// Add bias kernel with broadcasting: output[n,c,h,w] = input[n,c,h,w] + bias[c].
__global__ void add_bias_kernel(float* output, const float* input, const float* bias, int C, int HW, long long total_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < total_elements; i += stride) {
        int c = (int)((i / HW) % C);
        output[i] = input[i] + bias[c];
    }
}

// Conv2d forward kernel: each thread computes one output element.
// Assumes input, weight, bias are contiguous float32 tensors.
// input shape: [N, C, H, W], weight: [K, C, R, S], bias: [K]
// stride and padding are passed, dilation=1, groups=1.
__global__ void conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int stride, int padding,
    int H_out, int W_out
) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * K * H_out * W_out;
    long long grid_stride = (long long)blockDim.x * gridDim.x;

    for (long long linear = idx; linear < total; linear += grid_stride) {
        int w_out = linear % W_out;
        int h_out = (linear / W_out) % H_out;
        int k = (linear / ((long long)W_out * H_out)) % K;
        int n = linear / ((long long)W_out * H_out * K);

        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
                int h_in = h_out * stride - padding + r;
                if (h_in < 0 || h_in >= H) continue;
                for (int s = 0; s < S; ++s) {
                    int w_in = w_out * stride - padding + s;
                    if (w_in < 0 || w_in >= W) continue;
                    int input_idx = ((n * C + c) * H + h_in) * W + w_in;
                    int weight_idx = ((k * C + c) * R + r) * S + s;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
        output[linear] = sum + bias[k];
    }
}

// Launcher for conv2d_forward.
extern "C" void conv2d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int stride, int padding,
    int H_out, int W_out,
    cudaStream_t stream
) {
    const int block_size = 256;
    long long total = (long long)N * K * H_out * W_out;
    long long blocks = (total + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;
    conv2d_kernel<<<(int)blocks, block_size, 0, stream>>>(
        input, weight, bias, output,
        N, C, H, W, K, R, S, stride, padding, H_out, W_out
    );
}

// Launcher for leaky_relu_forward.
extern "C" void leaky_relu_launcher(float* output, const float* input, float negative_slope, long long total_elements, cudaStream_t stream) {
    const int block_size = 256;
    long long blocks = (total_elements + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;
    leaky_relu_kernel<<<(int)blocks, block_size, 0, stream>>>(output, input, negative_slope, total_elements);
}

// Launcher for add_bias_forward.
extern "C" void add_bias_launcher(float* output, const float* input, const float* bias, int C, int HW, long long total_elements, cudaStream_t stream) {
    const int block_size = 256;
    long long blocks = (total_elements + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535;
    add_bias_kernel<<<(int)blocks, block_size, 0, stream>>>(output, input, bias, C, HW, total_elements);
}
