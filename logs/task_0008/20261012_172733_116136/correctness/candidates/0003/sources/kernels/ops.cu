#include <cuda_runtime.h>
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// ------------------------- Pad Constant 2D -------------------------
__global__ void pad_constant_2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch, int channels, int in_h, int in_w,
    int pad_top, int pad_bottom, int pad_left, int pad_right,
    float pad_value) {
    int out_h = in_h + pad_top + pad_bottom;
    int out_w = in_w + pad_left + pad_right;
    int64_t total = (int64_t)batch * channels * out_h * out_w;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int w = idx % out_w;
        int tmp = idx / out_w;
        int h = tmp % out_h;
        tmp /= out_h;
        int c = tmp % channels;
        int b = tmp / channels;
        int in_h_idx = h - pad_top;
        int in_w_idx = w - pad_left;
        if (in_h_idx >= 0 && in_h_idx < in_h && in_w_idx >= 0 && in_w_idx < in_w) {
            output[idx] = input[((b * channels + c) * in_h + in_h_idx) * in_w + in_w_idx];
        } else {
            output[idx] = pad_value;
        }
    }
}

extern "C" void pad_constant_2d_launcher(
    const float* input, float* output,
    int batch, int channels, int in_h, int in_w,
    int pad_top, int pad_bottom, int pad_left, int pad_right,
    float pad_value, cudaStream_t stream) {
    int out_h = in_h + pad_top + pad_bottom;
    int out_w = in_w + pad_left + pad_right;
    int64_t total = (int64_t)batch * channels * out_h * out_w;
    int threads = 256;
    int blocks = (int)std::min((int64_t)((total + threads - 1) / threads), (int64_t)1048576);
    if (blocks == 0) return;
    pad_constant_2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch, channels, in_h, in_w,
        pad_top, pad_bottom, pad_left, pad_right, pad_value);
}

// ------------------------- Rot90 k=1 dims=(2,3) -------------------------
__global__ void rot90_k1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch, int channels, int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * height * width;
    if (idx >= total) return;
    int w = idx % width;
    int tmp = idx / width;
    int h = tmp % height;
    tmp /= height;
    int c = tmp % channels;
    int b = tmp / channels;
    int new_h = width;
    int new_w = height;
    int out_i = w;
    int out_j = height - 1 - h;
    int out_idx = ((b * channels + c) * new_h + out_i) * new_w + out_j;
    output[out_idx] = input[idx];
}

extern "C" void rot90_k1_launcher(
    const float* input, float* output,
    int batch, int channels, int height, int width,
    cudaStream_t stream) {
    int total = batch * channels * height * width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    rot90_k1_kernel<<<blocks, threads, 0, stream>>>(input, output, batch, channels, height, width);
}

// ------------------------- Conv2d forward -------------------------
__global__ void conv2d_direct_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int OH, int OW) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int nk = blockIdx.z;
    if (ow >= OW || oh >= OH) return;
    int n = nk / K;
    int k = nk % K;
    if (n >= N) return;
    float sum = bias[k];
    for (int c = 0; c < C; ++c) {
        for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
                int in_row = oh + r;
                int in_col = ow + s;
                float in_val = input[((n * C + c) * H + in_row) * W + in_col];
                float w_val = weight[((k * C + c) * R + r) * S + s];
                sum += in_val * w_val;
            }
        }
    }
    int out_idx = ((n * K + k) * OH + oh) * OW + ow;
    output[out_idx] = sum;
}

extern "C" void conv2d_forward_launcher(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C, int H, int W, int K, int R, int S,
    int OH, int OW, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((OW + block.x - 1) / block.x, (OH + block.y - 1) / block.y, N * K);
    conv2d_direct_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output, N, C, H, W, K, R, S, OH, OW);
}

// ------------------------- IsInf -------------------------
__global__ void isinf_kernel(const float* __restrict__ input, bool* __restrict__ output, int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        output[idx] = isinf(input[idx]);
    }
}

extern "C" void isinf_launcher(const float* input, bool* output, int64_t total, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)std::min((int64_t)((total + threads - 1) / threads), (int64_t)1048576);
    if (blocks == 0) return;
    isinf_kernel<<<blocks, threads, 0, stream>>>(input, output, total);
}

// ------------------------- Masked Fill Zero -------------------------
__global__ void masked_fill_zero_kernel(float* __restrict__ data, const bool* __restrict__ mask, int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        if (mask[idx]) {
            data[idx] = 0.0f;
        }
    }
}

extern "C" void masked_fill_zero_launcher(float* data, const bool* mask, int64_t total, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)std::min((int64_t)((total + threads - 1) / threads), (int64_t)1048576);
    if (blocks == 0) return;
    masked_fill_zero_kernel<<<blocks, threads, 0, stream>>>(data, mask, total);
}

// ------------------------- Add Bias -------------------------
__global__ void add_bias_kernel(float* __restrict__ data, const float* __restrict__ bias, int K, int H, int W, int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int w = idx % W;
        int64_t tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int k = tmp % K;
        data[idx] += bias[k];
    }
}

extern "C" void add_bias_launcher(float* data, const float* bias, int K, int H, int W, int64_t total, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)std::min((int64_t)((total + threads - 1) / threads), (int64_t)1048576);
    if (blocks == 0) return;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(data, bias, K, H, W, total);
}
