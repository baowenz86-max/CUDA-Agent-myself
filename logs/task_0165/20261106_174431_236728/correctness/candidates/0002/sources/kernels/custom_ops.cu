#include <cuda_runtime.h>

// log_softmax kernel
// ... (保持不变，但为了完整性放在这里)

// ConvTranspose1d kernel: output[n, co, l] = bias[co] + sum_{ci, k} input[n, ci, l*stride - padding + k] * weight[ci, co, k]
// where l*stride - padding + k is valid input index (>=0 && < L_in)
template <typename scalar_t>
__global__ void conv_transpose1d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int N, int C_in, int L_in, int C_out, int K, int stride, int padding, bool has_bias)
{
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    int co = blockIdx.z;
    if (l >= L_out || n >= N || co >= C_out) return;
    int L_out = (L_in - 1) * stride - 2 * padding + K; // assuming output_padding=0 for 1d
    scalar_t sum = has_bias ? bias[co] : 0;
    for (int ci = 0; ci < C_in; ++ci) {
        for (int k = 0; k < K; ++k) {
            int in_l = l * stride - padding + k;
            if (in_l >= 0 && in_l < L_in) {
                int in_idx = ((n * C_in + ci) * L_in + in_l);
                int w_idx = (ci * C_out + co) * K + k;
                sum += input[in_idx] * weight[w_idx];
            }
        }
    }
    output[((n * C_out + co) * L_out + l)] = sum;
}

// ConvTranspose2d kernel
template <typename scalar_t>
__global__ void conv_transpose2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ output,
    int N, int C_in, int H_in, int W_in, int C_out, int K_h, int K_w,
    int stride, int padding, int output_padding, bool has_bias)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y;
    int n = blockIdx.z / C_out;
    int co = blockIdx.z % C_out;
    int H_out = (H_in - 1) * stride - 2 * padding + K_h + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + K_w + output_padding;
    if (w >= W_out || h >= H_out || n >= N || co >= C_out) return;
    scalar_t sum = has_bias ? bias[co] : 0;
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                int in_h = h * stride - padding + kh;
                int in_w = w * stride - padding + kw;
                if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {
                    int in_idx = ((n * C_in + ci) * H_in + in_h) * W_in + in_w;
                    int w_idx = ((ci * C_out + co) * K_h + kh) * K_w + kw;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    output[((n * C_out + co) * H_out + h) * W_out + w] = sum;
}

// Launchers for conv_transpose1d/2d
extern "C" void conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int L_in, int C_out, int K,
    int stride, int padding, bool has_bias,
    cudaStream_t stream)
{
    int L_out = (L_in - 1) * stride - 2 * padding + K;
    dim3 block(256);
    dim3 grid((L_out + block.x - 1) / block.x, N, C_out);
    conv_transpose1d_kernel<float><<<grid, block, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, L_in, C_out, K, stride, padding, has_bias);
}

extern "C" void conv_transpose2d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in, int C_out, int K_h, int K_w,
    int stride, int padding, int output_padding, bool has_bias,
    cudaStream_t stream)
{
    int H_out = (H_in - 1) * stride - 2 * padding + K_h + output_padding;
    int W_out = (W_in - 1) * stride - 2 * padding + K_w + output_padding;
    dim3 block(256);
    dim3 grid((W_out + block.x - 1) / block.x, H_out, N * C_out);
    conv_transpose2d_kernel<float><<<grid, block, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, H_in, W_in, C_out, K_h, K_w,
        stride, padding, output_padding, has_bias);
}

// 原有的 log_softmax 代码保留（已存在）
// 注意：为了简洁，仅添加了新的函数，原有 log_softmax kernel 和 launcher 未改变，但需要一起放到文件里。
// 下面是完整文件内容（含原有 log_softmax 和新增部分）