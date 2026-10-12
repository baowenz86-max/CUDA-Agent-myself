#include <cuda_runtime.h>

// Batchnorm eval kernel: y = (x - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
__global__ void batchnorm_eval_kernel(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N,      // batch size
    int C,      // number of channels
    int HW      // height * width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    if (idx >= total) return;

    int c = (idx / HW) % C;
    float inv_std = rsqrtf(running_var[c] + eps);
    float value = (input[idx] - running_mean[c]) * inv_std * weight[c] + bias[c];
    output[idx] = value;
}

// BMM kernel: output[b, m, n] = sum_k mat1[b, m, k] * mat2[b, k, n]
__global__ void bmm_kernel(
    float* output,
    const float* mat1,
    const float* mat2,
    int batch,
    int M,
    int N,
    int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * M * N;
    if (idx >= total) return;

    int n = idx % N;
    int m = (idx / N) % M;
    int b = idx / (M * N);

    float sum = 0.0f;
    const float* a_row = mat1 + b * M * K + m * K;
    const float* b_col = mat2 + b * K * N + n;
    for (int k = 0; k < K; ++k) {
        sum += a_row[k] * b_col[k * N];
    }
    output[idx] = sum;
}

// ConvTranspose2d kernel for input shape [batch, in_channels, 1, seq_len]
// output shape [batch, out_channels, out_h, out_w]
// General transposed convolution: output[n, oc, oh, ow] = bias[oc] +
//   sum_{ic, kh, kw} input[n, ic, ih, iw] * weight[ic, oc, kh, kw]
// where oh = ih * stride_h - padding_h + kh, ow = iw * stride_w - padding_w + kw
// For our specific case: input height=1, width=seq_len.
__global__ void conv_transpose2d_kernel(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int batch,
    int in_channels,
    int out_channels,
    int kernel_H,
    int kernel_W,
    int stride_H,
    int stride_W,
    int padding_H,
    int padding_W,
    int output_padding_H,  // unused but kept for signature completeness
    int output_padding_W,  // unused
    int seq_len,           // input width
    int out_h,
    int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int oc = (idx / (out_w * out_h)) % out_channels;
    int n = idx / (out_w * out_h * out_channels);

    float acc = bias[oc];

    // Input height is 1, but we can generalize with ih loop (0 only here)
    for (int ic = 0; ic < in_channels; ++ic) {
        for (int kh = 0; kh < kernel_H; ++kh) {
            for (int kw = 0; kw < kernel_W; ++kw) {
                // Reverse mapping: input position = (output_position + padding - kernel_index) / stride
                int ih = oh + padding_H - kh;
                int iw = ow + padding_W - kw;
                if (ih % stride_H == 0 && iw % stride_W == 0) {
                    ih /= stride_H;
                    iw /= stride_W;
                    if (ih >= 0 && ih < 1 && iw >= 0 && iw < seq_len) {
                        // input index: n, ic, 0, iw
                        int input_idx = ((n * in_channels + ic) * 1 + 0) * seq_len + iw;
                        // weight index: ic, oc, kh, kw
                        int weight_idx = ((ic * out_channels + oc) * kernel_H + kh) * kernel_W + kw;
                        acc += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    output[idx] = acc;
}

// Launchers with extern C linkage for binding.cpp
extern "C" void batchnorm_eval_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N,
    int C,
    int HW,
    cudaStream_t stream
) {
    int total = N * C * HW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    batchnorm_eval_kernel<<<blocks, threads, 0, stream>>>(
        output, input, weight, bias, running_mean, running_var, eps, N, C, HW);
}

extern "C" void bmm_launcher(
    float* output,
    const float* mat1,
    const float* mat2,
    int batch,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    int total = batch * M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    bmm_kernel<<<blocks, threads, 0, stream>>>(
        output, mat1, mat2, batch, M, N, K);
}

extern "C" void conv_transpose2d_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int batch,
    int in_channels,
    int out_channels,
    int kernel_H,
    int kernel_W,
    int stride_H,
    int stride_W,
    int padding_H,
    int padding_W,
    int output_padding_H,
    int output_padding_W,
    int seq_len,
    cudaStream_t stream
) {
    // Compute output dimensions from input shape [batch, in_channels, 1, seq_len]
    int input_h = 1;
    int input_w = seq_len;
    int out_h = (input_h - 1) * stride_H - 2 * padding_H + kernel_H + output_padding_H;
    int out_w = (input_w - 1) * stride_W - 2 * padding_W + kernel_W + output_padding_W;
    int total = batch * out_channels * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv_transpose2d_kernel<<<blocks, threads, 0, stream>>>(
        output, input, weight, bias,
        batch, in_channels, out_channels,
        kernel_H, kernel_W, stride_H, stride_W,
        padding_H, padding_W, output_padding_H, output_padding_W,
        seq_len, out_h, out_w);
}
