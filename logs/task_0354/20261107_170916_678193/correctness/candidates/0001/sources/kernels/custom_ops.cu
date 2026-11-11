#include <cuda_runtime.h>

__global__ void replication_pad2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H_in, int W_in,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int H_out, int W_out,
    int64_t total_elements)
{
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t i = idx; i < total_elements; i += stride) {
        int w_out = i % W_out;
        int64_t tmp = i / W_out;
        int h_out = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        int h_src = h_out - pad_top;
        h_src = h_src < 0 ? 0 : h_src;
        h_src = h_src >= H_in ? H_in - 1 : h_src;

        int w_src = w_out - pad_left;
        w_src = w_src < 0 ? 0 : w_src;
        w_src = w_src >= W_in ? W_in - 1 : w_src;

        int64_t input_idx = (((int64_t)n * C + c) * H_in + h_src) * W_in + w_src;
        output[i] = input[input_idx];
    }
}

extern "C" void replication_pad2d_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H_in, int W_in,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    cudaStream_t stream)
{
    int H_out = H_in + pad_top + pad_bottom;
    int W_out = W_in + pad_left + pad_right;
    int64_t total_elements = (int64_t)N * C * H_out * W_out;

    if (total_elements == 0) return;

    const int block_size = 256;
    int64_t blocks = (total_elements + block_size - 1) / block_size;
    if (blocks > 2147483647) blocks = 2147483647;

    replication_pad2d_kernel<<<(int)blocks, block_size, 0, stream>>>(
        output, input, N, C, H_in, W_in,
        pad_left, pad_right, pad_top, pad_bottom,
        H_out, W_out, total_elements);
}
