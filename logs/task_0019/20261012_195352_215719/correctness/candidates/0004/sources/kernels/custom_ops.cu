#include <cuda_runtime.h>

// Helper device functions
__device__ __forceinline__ float clamp_scalar(float x, float lo, float hi) {
    return fminf(fmaxf(x, lo), hi);
}

__device__ __forceinline__ float leaky_relu_scalar(float x, float slope) {
    return (x >= 0.0f) ? x : slope * x;
}

// ===================== ConvTranspose2d =====================
// Parameters fixed from model.py: kernel_size=3, stride=2, padding=1, output_padding=1
// General kernel supports arbitrary parameters via launcher args.
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w
) {
    int total = N * C_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int w = i % W_out;
        int h = (i / W_out) % H_out;
        int c_out = (i / (W_out * H_out)) % C_out;
        int n = i / (W_out * H_out * C_out);

        float acc = 0.0f;
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int h_in = h + padding_h - kh;
                    int w_in = w + padding_w - kw;
                    if (h_in % stride_h != 0 || w_in % stride_w != 0) continue;
                    h_in /= stride_h;
                    w_in /= stride_w;
                    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
                        // weight is stored as [C_in][C_out][kh][kw] for ConvTranspose2d
                        int w_idx = ((c_in * C_out + c_out) * kernel_h + kh) * kernel_w + kw;
                        int in_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                        acc += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
        if (bias != nullptr) {
            acc += bias[c_out];
        }
        output[i] = acc;
    }
}

extern "C" void conv_transpose2d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    cudaStream_t stream
) {
    int total = N * C_out * H_out * W_out;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Cap grids to avoid exceeding max grid size, use grid-stride loop inside kernel
    if (blocks > 65535) blocks = 65535;
    conv_transpose2d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        output_padding_h, output_padding_w
    );
}

// ===================== Clamp =====================
__global__ void clamp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float min_val, float max_val,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        output[i] = clamp_scalar(input[i], min_val, max_val);
    }
}

extern "C" void clamp_forward_launcher(
    const float* input,
    float* output,
    float min_val, float max_val,
    int total,
    cudaStream_t stream
) {
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    clamp_kernel<<<blocks, threads, 0, stream>>>(
        input, output, min_val, max_val, total
    );
}

// ===================== Leaky ReLU =====================
__global__ void leaky_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float slope,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        output[i] = leaky_relu_scalar(input[i], slope);
    }
}

extern "C" void leaky_relu_forward_launcher(
    const float* input,
    float* output,
    float slope,
    int total,
    cudaStream_t stream
) {
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    leaky_relu_kernel<<<blocks, threads, 0, stream>>>(
        input, output, slope, total
    );
}

// ===================== Triu =====================
// triu with diagonal=0; zero elements below main diagonal in last two dims (H, W)
__global__ void triu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int diagonal
) {
    int total = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int w = i % W;
        int h = (i / W) % H;
        // batch and channel indices not needed for decision
        if (w - h < diagonal) {
            output[i] = 0.0f;
        } else {
            output[i] = input[i];
        }
    }
}

extern "C" void triu_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int diagonal,
    cudaStream_t stream
) {
    int total = N * C * H * W;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    triu_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, diagonal
    );
}

// ===================== Add (bias shape [C,1,1]) =====================
__global__ void add_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W
) {
    int total = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int w = i % W;
        int h = (i / W) % H;
        int c = (i / (W * H)) % C;
        output[i] = input[i] + bias[c];
    }
}

extern "C" void add_forward_launcher(
    const float* input,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    cudaStream_t stream
) {
    int total = N * C * H * W;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(
        input, bias, output, N, C, H, W
    );
}
