#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// triu_forward kernel
// Applies torch.triu (diagonal=0) to each HxW matrix in the last two dimensions.
// Input and output have shape [N, C, H, W], potentially non-contiguous.
// Each thread handles one element of the output.
// -----------------------------------------------------------------------------
__global__ void triu_kernel_grid_stride(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long total,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;

        float val = 0.0f;
        if (h <= w) {
            const float* in_ptr = input + n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w;
            val = *in_ptr;
        }
        float* out_ptr = output + n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w;
        *out_ptr = val;
    }
}

extern "C" void triu_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H * W;
    if (total <= 0) return;
    const int threads = 256;
    int grid = (int)min((long long)65535, (total + threads - 1) / threads);
    if (grid == 0) grid = 1;
    triu_kernel_grid_stride<<<grid, threads, 0, stream>>>(
        output, input, total, N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w
    );
}

// -----------------------------------------------------------------------------
// add_bias_forward kernel
// Adds bias of shape [C] or [C,1,1] to x of shape [N,C,H,W] following broadcasting.
// We assume bias is contiguous and of shape [C] or [C,1,1] with C elements.
// -----------------------------------------------------------------------------
__global__ void add_bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ bias,
    long long total,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;

        float b = bias[c];
        float val = input[n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w];
        output[n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w] = val + b;
    }
}

extern "C" void add_bias_forward_launcher(
    float* output,
    const float* input,
    const float* bias,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    long long total = (long long)N * C * H * W;
    if (total <= 0) return;
    const int threads = 256;
    int grid = (int)min((long long)65535, (total + threads - 1) / threads);
    if (grid == 0) grid = 1;
    add_bias_kernel<<<grid, threads, 0, stream>>>(
        output, input, bias, total, N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w
    );
}
