#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// triu_forward kernel
// Applies torch.triu (diagonal=0) to each HxW matrix in the last two dimensions.
// Input and output have shape [N, C, H, W], potentially non-contiguous.
// Each thread handles one element of the output.
// -----------------------------------------------------------------------------
__global__ void triu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * C * H * W;
    if (idx >= total) return;

    // Decode linear index to n,c,h,w
    int w = idx % W;
    long long tmp = idx / W;
    int h = tmp % H;
    tmp /= H;
    int c = tmp % C;
    int n = tmp / C;

    // triu condition: keep if h <= w (i.e., on or above main diagonal)
    float val = 0.0f;
    if (h <= w) {
        const float* in_ptr = input + n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w;
        val = *in_ptr;
    }
    float* out_ptr = output + n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w;
    *out_ptr = val;
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
    long long blocks = (total + threads - 1) / threads;
    // Cap grid dimension at 2^31-1 and use grid-stride loop if needed.
    // For simplicity, if blocks exceed max grid size, use a grid-stride loop version.
    // Here we assume typical sizes fit; implement a safe cap with loop.
    int max_blocks = 65535; // conservative for older GPUs; can be larger but safe
    int grid = (int)min((long long)max_blocks, blocks);
    triu_kernel<<<grid, threads, 0, stream>>>(
        output, input, N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w
    );
    // If blocks > max_blocks, we need a grid-stride loop; but for our target sizes (512*3*64*64=6M),
    // blocks=24576 which is below 65535, so it's fine. However, include loop to be safe.
}

// Grid-stride loop variant for very large tensors (not actually needed for current task)
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

// We'll use the grid-stride version always for safety and simplicity.
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
    // Cap grid at a reasonable number, then use grid-stride loop in kernel.
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
// Output shape = input shape, contiguous.
// -----------------------------------------------------------------------------
__global__ void add_bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ bias,
    long long total,
    int C, int HW,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride) {
        // Decode n,c,h,w from idx: idx = n*C*HW + c*HW + hw
        int hw = idx % HW;
        long long tmp = idx / HW;
        int c = tmp % C;
        // For bias shape [C,1,1] or [C], bias index is just c
        float b = bias[c];
        // Compute input value using strides
        long long n = tmp / C;
        // Need h,w from hw
        int h = hw / (int)out_stride_h; // unsafe; better pass H,W separately. We'll assume output contiguous: out_stride_h = W, out_stride_w=1. For simplicity but correctness.
        // Actually we should pass H,W to decode; we'll do that in final version below.
        // Output pointer: output + n*out_stride_n + c*out_stride_c + h*out_stride_h + w*out_stride_w
        // Since output is contiguous, this simplifies; but to be general, use n,c,h,w.
    }
}

// Proper implementation with H,W passed.
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
