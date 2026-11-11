#include <cuda_runtime.h>

// custom_pad2d kernel
// Applies ZeroPad2d with given padding to input tensor of shape [N, C, H, W]
// Output is contiguous of shape [N, C, H_out, W_out], where
// H_out = H + pad_top + pad_bottom, W_out = W + pad_left + pad_right
// Input strides are used to support non-contiguous input.
__global__ void custom_pad2d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int pad_left, int pad_top,
    long long input_n_stride, long long input_c_stride,
    long long input_h_stride, long long input_w_stride)
{
    long long total = (long long)N * C * H_out * W_out;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int w_out = (int)(idx % W_out);
        int h_out = (int)((idx / W_out) % H_out);
        int c = (int)((idx / ((long long)H_out * W_out)) % C);
        int n = (int)(idx / ((long long)C * H_out * W_out));

        int h_in = h_out - pad_top;
        int w_in = w_out - pad_left;

        float val = 0.0f;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
            const float* src = input + n * input_n_stride + c * input_c_stride
                               + h_in * input_h_stride + w_in * input_w_stride;
            val = *src;
        }
        output[idx] = val;
    }
}

// custom_mean_dim kernel specialized for dim = 2 (height), input contiguous [N, C, H, W]
// Output contiguous [N, C, W]
__global__ void custom_mean_dim2_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    float inv_H)
{
    int total = N * C * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int w = idx % W;
        int c = (idx / W) % C;
        int n = idx / (C * W);

        const float* src = input + (long long)n * C * H * W + c * H * W + w;
        float sum = 0.0f;
        #pragma unroll 4
        for (int h = 0; h < H; ++h) {
            sum += src[(long long)h * W];
        }
        output[idx] = sum * inv_H;
    }
}

// Launcher for custom_pad2d
// All parameters are raw pointers/scalars, no PyTorch dependencies.
// Input strides are passed as long long (bytes are not needed because element strides are used).
extern "C" void custom_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    long long input_n_stride, long long input_c_stride,
    long long input_h_stride, long long input_w_stride,
    cudaStream_t stream)
{
    int H_out = H + pad_top + pad_bottom;
    int W_out = W + pad_left + pad_right;
    long long total = (long long)N * C * H_out * W_out;

    const int block = 256;
    long long grid_ll = (total + block - 1) / block;
    // gridDim.x is int, so cap at 2^31-1; but total for this task is small enough.
    int grid = (int)grid_ll;
    if (grid < 1) grid = 1;

    custom_pad2d_kernel<<<grid, block, 0, stream>>>(
        output, input,
        N, C, H, W,
        H_out, W_out,
        pad_left, pad_top,
        input_n_stride, input_c_stride,
        input_h_stride, input_w_stride);
}

// Launcher for custom_mean_dim, restricted to dim=2 based on contract.
// input is assumed contiguous (as produced by custom_pad2d).
extern "C" void custom_mean_dim_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int dim,
    cudaStream_t stream)
{
    if (dim != 2) {
        // For safety, if dim != 2, do nothing? Contract only uses dim=2.
        // We'll still handle only dim=2; caller must ensure dim=2.
        return;
    }

    int total = N * C * W;
    int block = 256;
    int grid = (total + block - 1) / block;
    if (grid < 1) grid = 1;

    float inv_H = 1.0f / (float)H;
    custom_mean_dim2_kernel<<<grid, block, 0, stream>>>(
        output, input, N, C, H, W, inv_H);
}
