#include <cuda_runtime.h>

// ------------------------- ConvTranspose2d Kernel -------------------------
// Naive direct convolution for output = input * weight^T + bias
// Assumes float32, contiguous tensors, stride=2, padding=1, output_padding=1, kernel_size=3
// Output dimensions: outH = (inH-1)*stride - 2*padding + kernel_size + output_padding
//                   outW = (inW-1)*stride - 2*padding + kernel_size + output_padding
__global__ void conv_transpose2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w
) {
    int total = N * C_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_total = blockDim.x * gridDim.x;

    for (int pos = idx; pos < total; pos += stride_total) {
        int w_out = pos % W_out;
        int tmp = pos / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c_out = tmp % C_out;
        int n = tmp / C_out;

        float val = (bias != nullptr) ? bias[c_out] : 0.0f;

        // Iterate over input channels and kernel positions
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                // h_in such that h_out = h_in*stride_h - padding_h + kh
                int h_in = h_out + padding_h - kh;
                if (h_in < 0 || h_in % stride_h != 0) continue;
                h_in /= stride_h;
                if (h_in >= H_in) continue;

                for (int kw = 0; kw < kernel_w; ++kw) {
                    int w_in = w_out + padding_w - kw;
                    if (w_in < 0 || w_in % stride_w != 0) continue;
                    w_in /= stride_w;
                    if (w_in >= W_in) continue;

                    // weight shape: [C_in, C_out, kernel_h, kernel_w]
                    int w_idx = ((c_in * C_out + c_out) * kernel_h + kh) * kernel_w + kw;
                    int i_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    val += input[i_idx] * weight[w_idx];
                }
            }
        }
        output[pos] = val;
    }
}

extern "C" void conv_transpose_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    cudaStream_t stream
) {
    int total = N * C_out * H_out * W_out;
    int threads = 256;
    // Cap grid at 65535 and use grid-stride loop above
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    conv_transpose2d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, H_in, W_in, C_out, H_out, W_out,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w
    );
}

// ------------------------- Clamp Kernel -------------------------
__global__ void clamp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total,
    float min_val,
    float max_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        float x = input[i];
        x = fminf(fmaxf(x, min_val), max_val);
        output[i] = x;
    }
}

extern "C" void clamp_forward_launcher(
    const float* input,
    float* output,
    int total,
    float min_val,
    float max_val,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    clamp_kernel<<<blocks, threads, 0, stream>>>(input, output, total, min_val, max_val);
}

// ------------------------- Leaky ReLU Kernel -------------------------
__global__ void leaky_relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total,
    float negative_slope
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        float x = input[i];
        output[i] = (x > 0.0f) ? x : (x * negative_slope);
    }
}

extern "C" void leaky_relu_forward_launcher(
    const float* input,
    float* output,
    int total,
    float negative_slope,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    leaky_relu_kernel<<<blocks, threads, 0, stream>>>(input, output, total, negative_slope);
}

// ------------------------- Triu Kernel -------------------------
// Applies upper triangular mask to last two dimensions (H_out x W_out)
// diagonal=0 so keep elements where col >= row
__global__ void triu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int diagonal
) {
    int total = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int pos = idx; pos < total; pos += stride) {
        int w = pos % W;
        int tmp = pos / W;
        int h = tmp % H;
        // Remaining dims (C, N) are not needed for indexing input/output
        output[pos] = (w >= h + diagonal) ? input[pos] : 0.0f;
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
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    triu_kernel<<<blocks, threads, 0, stream>>>(input, output, N, C, H, W, diagonal);
}

// ------------------------- Add Kernel (broadcast bias) -------------------------
// Adds bias of shape (C_out,1,1) to input of shape (N,C_out,H,W)
__global__ void add_bias_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W
) {
    int total = N * C * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int pos = idx; pos < total; pos += stride) {
        int w = pos % W;
        int tmp = pos / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        output[pos] = input[pos] + bias[c];
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
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(input, bias, output, N, C, H, W);
}
