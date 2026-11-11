#include <cuda_runtime.h>

// Kernel for reflection padding
__global__ void reflection_pad2d_float_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int pad_h, int pad_w,
    long long in_stride_n, long long in_stride_c,
    long long in_stride_h, long long in_stride_w)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = (int64_t)N * C * H_out * W_out;
    if (idx >= total_elements) return;

    int w_out = idx % W_out;
    int64_t tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;

    int h_in = h_out - pad_h;
    if (h_in < 0) h_in = -h_in;
    if (h_in >= H_in) h_in = 2 * (H_in - 1) - h_in;

    int w_in = w_out - pad_w;
    if (w_in < 0) w_in = -w_in;
    if (w_in >= W_in) w_in = 2 * (W_in - 1) - w_in;

    long long input_offset = (long long)n * in_stride_n
                           + (long long)c * in_stride_c
                           + (long long)h_in * in_stride_h
                           + (long long)w_in * in_stride_w;

    output[idx] = input[input_offset];
}

// Kernel for gather along dim=1
__global__ void gather_dim1_float_kernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ indices,
    float* __restrict__ output,
    int N, int C_x, int C_out, int H, int W,
    long long in_stride_n, long long in_stride_c,
    long long in_stride_h, long long in_stride_w,
    long long idx_stride_n, long long idx_stride_c,
    long long idx_stride_h, long long idx_stride_w)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = (int64_t)N * C_out * H * W;
    if (idx >= total_elements) return;

    int w = idx % W;
    int64_t tmp = idx / W;
    int h = tmp % H;
    tmp /= H;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    long long idx_offset = (long long)n * idx_stride_n
                         + (long long)c_out * idx_stride_c
                         + (long long)h * idx_stride_h
                         + (long long)w * idx_stride_w;
    int c_in = (int)indices[idx_offset];

    long long in_offset = (long long)n * in_stride_n
                        + (long long)c_in * in_stride_c
                        + (long long)h * in_stride_h
                        + (long long)w * in_stride_w;

    output[idx] = input[in_offset];
}

// Kernel for 2D convolution with bias (NCHW, float32)
// weight shape: [C_out, C_in, K_h, K_w]
// input shape: [N, C_in, H_in, W_in]
// output shape: [N, C_out, H_out, W_out]
// stride = 1, padding = 0 (as required by task)
__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int H_out, int W_out)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = (int64_t)N * C_out * H_out * W_out;
    if (idx >= total_elements) return;

    int w_out = idx % W_out;
    int64_t tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c_out = tmp % C_out;
    int n = tmp / C_out;

    float sum = bias[c_out];
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kh = 0; kh < K_h; ++kh) {
            for (int kw = 0; kw < K_w; ++kw) {
                int h_in = h_out + kh;
                int w_in = w_out + kw;
                // No padding, so h_in, w_in are within bounds
                sum += input[((n * C_in + c_in) * H_in + h_in) * W_in + w_in] *
                       weight[((c_out * C_in + c_in) * K_h + kh) * K_w + kw];
            }
        }
    }
    output[idx] = sum;
}

// Launcher for reflection_pad2d_forward
extern "C" void reflection_pad2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int padding,
    long long in_stride_n, long long in_stride_c,
    long long in_stride_h, long long in_stride_w,
    cudaStream_t stream)
{
    int H_out = H_in + 2 * padding;
    int W_out = W_in + 2 * padding;
    int64_t total_elements = (int64_t)N * C * H_out * W_out;

    int block_size = 256;
    int64_t grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 0) {
        reflection_pad2d_float_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
            input, output, N, C, H_in, W_in, H_out, W_out, padding, padding,
            in_stride_n, in_stride_c, in_stride_h, in_stride_w);
    }
}

// Launcher for gather_custom_forward
extern "C" void gather_custom_forward_launcher(
    const float* input,
    const int64_t* indices,
    float* output,
    int N, int C_x, int C_out, int H, int W,
    long long in_stride_n, long long in_stride_c,
    long long in_stride_h, long long in_stride_w,
    long long idx_stride_n, long long idx_stride_c,
    long long idx_stride_h, long long idx_stride_w,
    cudaStream_t stream)
{
    int64_t total_elements = (int64_t)N * C_out * H * W;
    int block_size = 256;
    int64_t grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 0) {
        gather_dim1_float_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
            input, indices, output,
            N, C_x, C_out, H, W,
            in_stride_n, in_stride_c, in_stride_h, in_stride_w,
            idx_stride_n, idx_stride_c, idx_stride_h, idx_stride_w);
    }
}

// Launcher for conv2d forward (no padding, stride=1)
extern "C" void conv2d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int K_h, int K_w,
    int H_out, int W_out,
    cudaStream_t stream)
{
    int64_t total_elements = (int64_t)N * C_out * H_out * W_out;
    int block_size = 256;
    int64_t grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 0) {
        conv2d_forward_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
            input, weight, bias, output,
            N, C_in, H_in, W_in,
            C_out, K_h, K_w, H_out, W_out);
    }
}
