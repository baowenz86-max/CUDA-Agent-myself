#include <cuda_runtime.h>

// Elementwise fused sqrt+round kernel for float32 tensors.
// Input values are assumed non-negative (as per contract).
__global__ void fused_sqrt_round_kernel(float* __restrict__ output,
                                        const float* __restrict__ input,
                                        long long numel) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float val = input[i];
        // torch.sqrt then torch.round (round half to even)
        output[i] = roundf(sqrtf(val));
    }
}

// Host launcher for fused sqrt+round
extern "C" void fused_sqrt_round_launcher(float* output,
                                          const float* input,
                                          long long numel,
                                          cudaStream_t stream) {
    const int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    const long long max_blocks = 1048576;
    int blocks = (int)(blocks_ll < max_blocks ? blocks_ll : max_blocks);
    if (blocks == 0) return;
    fused_sqrt_round_kernel<<<blocks, threads, 0, stream>>>(output, input, numel);
}

// Custom transposed convolution kernel
// Assumes NCHW layout, float32, stride=2, padding=1, output_padding=1 (as per contract)
// But we generalize: kernel is parameterized by stride, padding, output_padding.
// Each thread computes one output element.
__global__ void conv_transpose2d_kernel(const float* __restrict__ input,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias,
                                        float* __restrict__ output,
                                        int N, int C_in, int H_in, int W_in,
                                        int C_out, int kH, int kW,
                                        int stride, int padding, int output_padding,
                                        int H_out, int W_out) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_size = N * C_out * H_out * W_out;
    if (out_idx >= out_size) return;

    // Decompose index
    int w_out = out_idx % W_out;
    int h_out = (out_idx / W_out) % H_out;
    int c_out = (out_idx / (W_out * H_out)) % C_out;
    int n = out_idx / (W_out * H_out * C_out);

    float acc = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                // Compute corresponding input position
                int h_in = (h_out - output_padding + 2 * padding - kh) / stride;
                int w_in = (w_out - output_padding + 2 * padding - kw) / stride;
                // Check if (h_out - output_padding + 2*padding - kh) is divisible by stride and within input bounds
                int numerator_h = h_out - output_padding + 2 * padding - kh;
                int numerator_w = w_out - output_padding + 2 * padding - kw;
                if (numerator_h % stride != 0 || numerator_w % stride != 0) continue;
                if (h_in < 0 || h_in >= H_in || w_in < 0 || w_in >= W_in) continue;
                // Access input[n, c_in, h_in, w_in]
                int in_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                float in_val = input[in_idx];
                // Access weight[c_in, c_out, kh, kw] (PyTorch ConvTranspose2d weight shape: [C_in, C_out, kH, kW])
                int weight_idx = ((c_in * C_out + c_out) * kH + kh) * kW + kw;
                float w_val = weight[weight_idx];
                acc += in_val * w_val;
            }
        }
    }
    // Add bias if present
    if (bias != nullptr) {
        acc += bias[c_out];
    }
    output[out_idx] = acc;
}

// Host launcher for conv_transpose2d
// Parameters: input, weight, bias (can be null), output, dimensions, convolution params, stream
extern "C" void conv_transpose2d_launcher(const float* input,
                                          const float* weight,
                                          const float* bias,
                                          float* output,
                                          int N, int C_in, int H_in, int W_in,
                                          int C_out, int kH, int kW,
                                          int stride, int padding, int output_padding,
                                          int H_out, int W_out,
                                          cudaStream_t stream) {
    int out_size = N * C_out * H_out * W_out;
    const int threads = 256;
    int blocks = (out_size + threads - 1) / threads;
    conv_transpose2d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, H_in, W_in,
        C_out, kH, kW,
        stride, padding, output_padding,
        H_out, W_out);
}
