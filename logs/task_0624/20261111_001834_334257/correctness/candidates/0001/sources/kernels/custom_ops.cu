#include <cuda_runtime.h>
#include <math.h>

// -----------------------------------------------------------------------------
// channel_sum: sum over dimension 1 (channels) for a 4D tensor.
// Input: x with shape [B, C, H, W] and strides [sB, sC, sH, sW].
// Output: y with shape [B, H, W] and strides [out_sB, out_sH, out_sW].
// For continuous input, sC >= sH >= sW, but kernel uses strides directly.
// Each thread handles one (b, h, w) position after grid-stride loop.
//
// Grid: blocks determined by total output elements, blockDim fixed 256.
// -----------------------------------------------------------------------------
__global__ void channel_sum_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    int B,
    int C,
    int H,
    int W,
    long long sB,  // use long long for potentially large strides (int may overflow for 512*512*3)
    long long sC,
    long long sH,
    long long sW,
    long long out_sB,
    long long out_sH,
    long long out_sW,
    long long total_out
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (; idx < total_out; idx += stride) {
        // Decompose idx into b, h, w.
        int w = idx % W;
        long long rem = idx / W;
        int h = rem % H;
        int b = rem / H;

        const float* base = x + b * sB + h * sH + w * sW;
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            sum += base[c * sC];
        }
        y[b * out_sB + h * out_sH + w * out_sW] = sum;
    }
}

// -----------------------------------------------------------------------------
// exp: element-wise natural exponential using __expf for speed.
// Input and output share shape and strides, but launcher uses actual pointers.
// -----------------------------------------------------------------------------
__global__ void exp_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    long long total
) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (; idx < total; idx += stride) {
        y[idx] = __expf(x[idx]);
    }
}

// -----------------------------------------------------------------------------
// Launchers (C interface for binding)
// -----------------------------------------------------------------------------
extern "C" void channel_sum_launcher(
    const float* x,
    float* y,
    int B,
    int C,
    int H,
    int W,
    long long sB,
    long long sC,
    long long sH,
    long long sW,
    long long out_sB,
    long long out_sH,
    long long out_sW,
    cudaStream_t stream
) {
    long long total_out = (long long)B * H * W;
    if (total_out <= 0) return;

    const int block = 256;
    int grid = (int)((total_out + block - 1) / block);
    if (grid > 1048576) grid = 1048576; // practical grid cap

    channel_sum_kernel<<<grid, block, 0, stream>>>(
        x, y, B, C, H, W,
        sB, sC, sH, sW,
        out_sB, out_sH, out_sW,
        total_out
    );
}

extern "C" void exp_launcher(
    const float* x,
    float* y,
    long long total,
    cudaStream_t stream
) {
    if (total <= 0) return;

    const int block = 256;
    int grid = (int)((total + block - 1) / block);
    if (grid > 1048576) grid = 1048576;

    exp_kernel<<<grid, block, 0, stream>>>(x, y, total);
}
