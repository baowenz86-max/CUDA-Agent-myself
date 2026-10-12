// custom_ops.cu
#include <cuda_runtime.h>

// Helper: clamp kernel
__global__ void clamp_kernel(float* out, const float* in, int n, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = in[idx];
        v = v < min_val ? min_val : v;
        v = v > max_val ? max_val : v;
        out[idx] = v;
    }
}

// Helper: leaky_relu kernel
__global__ void leaky_relu_kernel(float* out, const float* in, int n, float negative_slope) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = in[idx];
        out[idx] = v >= 0.0f ? v : v * negative_slope;
    }
}

// Helper: triu kernel (applied to last two dimensions of each sample)
__global__ void triu_kernel(float* out, const float* in, int total_elements, int H, int W, int diagonal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int w = idx % W;
    int h = (idx / W) % H;
    if (w >= h + diagonal) {
        out[idx] = in[idx];
    } else {
        out[idx] = 0.0f;
    }
}

// Helper: add bias (bias shape [C,1,1]) broadcast to [N,C,H,W]
__global__ void add_bias_kernel(float* out, const float* in, const float* bias, int N, int C, int HW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * HW) return;
    int c = (idx / HW) % C;
    out[idx] = in[idx] + bias[c];
}

// ConvTranspose2d kernel
// Weight layout: [in_channels, out_channels, kH, kW]
// Input: [N, C_in, H_in, W_in]
// Output: [N, C_out, H_out, W_out]
__global__ void conv_transpose2d_kernel(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kH, int kW,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C_out * H_out * W_out) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c_out = (idx / (W_out * H_out)) % C_out;
    int n = idx / (W_out * H_out * C_out);

    float val = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                int h_in = h_out + padding_h - kh;
                int w_in = w_out + padding_w - kw;
                if (h_in % stride_h != 0 || w_in % stride_w != 0) continue;
                h_in /= stride_h;
                w_in /= stride_w;
                if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                    float input_val = input[((n * C_in + c_in) * H_in + h_in) * W_in + w_in];
                    float weight_val = weight[((c_in * C_out + c_out) * kH + kh) * kW + kw];
                    val += input_val * weight_val;
                }
            }
        }
    }
    val += bias[c_out];
    output[idx] = val;
}

// Launchers
extern "C" void conv_transpose2d_forward_launcher(
    const float* input, const float* weight, const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kH, int kW,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    cudaStream_t stream
) {
    int total = N * C_out * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv_transpose2d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kH, kW, stride_h, stride_w, padding_h, padding_w,
        output_padding_h, output_padding_w
    );
}

extern "C" void clamp_forward_launcher(
    const float* input, float* output, int n, float min_val, float max_val,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    clamp_kernel<<<blocks, threads, 0, stream>>>(output, input, n, min_val, max_val);
}

extern "C" void leaky_relu_forward_launcher(
    const float* input, float* output, int n, float negative_slope,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads, 0, stream>>>(output, input, n, negative_slope);
}

extern "C" void triu_forward_launcher(
    const float* input, float* output, int total_elements, int H, int W, int diagonal,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    triu_kernel<<<blocks, threads, 0, stream>>>(output, input, total_elements, H, W, diagonal);
}

extern "C" void add_forward_launcher(
    const float* input, const float* bias, float* output,
    int N, int C, int HW,
    cudaStream_t stream
) {
    int total = N * C * HW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(output, input, bias, N, C, HW);
}
