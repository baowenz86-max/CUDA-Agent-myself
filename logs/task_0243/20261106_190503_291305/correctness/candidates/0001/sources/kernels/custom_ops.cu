#include <cuda_runtime.h>

// 3D average pooling kernel for kernel_size=(2,2,2), stride=(2,2,2), padding=0
// Input tensor shape: [N, C, D, H, W]
// Output tensor shape: [N, C, D_out, H_out, W_out]
// Output is assumed contiguous; input may have arbitrary strides.
__global__ void avg_pool3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int in_stride_n, int in_stride_c, int in_stride_d, int in_stride_h, int in_stride_w)
{
    int total_outputs = N * C * D_out * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int linear = idx; linear < total_outputs; linear += stride) {
        // Decompose linear index into n, c, od, oh, ow
        int ow = linear % W_out;
        int tmp = linear / W_out;
        int oh = tmp % H_out;
        tmp /= H_out;
        int od = tmp % D_out;
        tmp /= D_out;
        int c = tmp % C;
        int n = tmp / C;

        // Corresponding input start indices (stride 2, no padding)
        int in_d = od * 2;
        int in_h = oh * 2;
        int in_w = ow * 2;

        // Accumulate over the 2x2x2 window
        float sum = 0.0f;
        sum += input[n * in_stride_n + c * in_stride_c + in_d * in_stride_d + in_h * in_stride_h + in_w * in_stride_w];
        sum += input[n * in_stride_n + c * in_stride_c + in_d * in_stride_d + in_h * in_stride_h + (in_w + 1) * in_stride_w];
        sum += input[n * in_stride_n + c * in_stride_c + in_d * in_stride_d + (in_h + 1) * in_stride_h + in_w * in_stride_w];
        sum += input[n * in_stride_n + c * in_stride_c + in_d * in_stride_d + (in_h + 1) * in_stride_h + (in_w + 1) * in_stride_w];
        sum += input[n * in_stride_n + c * in_stride_c + (in_d + 1) * in_stride_d + in_h * in_stride_h + in_w * in_stride_w];
        sum += input[n * in_stride_n + c * in_stride_c + (in_d + 1) * in_stride_d + in_h * in_stride_h + (in_w + 1) * in_stride_w];
        sum += input[n * in_stride_n + c * in_stride_c + (in_d + 1) * in_stride_d + (in_h + 1) * in_stride_h + in_w * in_stride_w];
        sum += input[n * in_stride_n + c * in_stride_c + (in_d + 1) * in_stride_d + (in_h + 1) * in_stride_h + (in_w + 1) * in_stride_w];

        output[linear] = sum / 8.0f;
    }
}

extern "C" void avg_pool3d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int in_stride_n, int in_stride_c, int in_stride_d, int in_stride_h, int in_stride_w,
    cudaStream_t stream)
{
    int total_outputs = N * C * D_out * H_out * W_out;
    if (total_outputs <= 0) return;

    const int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    // Grid-stride loop handles any size; no need to cap blocks but keep reasonable.
    // Limit blocks to avoid excessive launch overhead for very large tensors.
    const int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;

    avg_pool3d_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, D, H, W,
        D_out, H_out, W_out,
        in_stride_n, in_stride_c, in_stride_d, in_stride_h, in_stride_w);
}
