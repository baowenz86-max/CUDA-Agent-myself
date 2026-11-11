#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// sum_channels_launcher: reduces over channel dimension (dim=1) with keepdim=true
// Input: x shape [N, C, H, W], strides [sN, sC, sH, sW] (arbitrary, but typically contiguous)
// Output: y shape [N, 1, H, W], contiguous (strides [H*W? Actually output is contiguous with shape [N,1,H,W] so strides [H*W, H*W, W, 1])
// We compute each output element y[n,0,h,w] = sum_c x[n,c,h,w]
// -----------------------------------------------------------------------------
template <typename T>
__global__ void sum_channels_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int N, int C, int H, int W,
    long long sN, long long sC, long long sH, long long sW,
    long long total_out)  // N*H*W
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < total_out; idx += stride) {
        int w = idx % W;
        int tmp = idx / W;
        int h = tmp % H;
        int n = tmp / H;
        // sum over c
        T sum = 0;
        const T* x_base = x + (long long)n * sN + (long long)h * sH + (long long)w * sW;
        for (int c = 0; c < C; ++c) {
            sum += x_base[c * sC];
        }
        y[idx] = sum;
    }
}

extern "C" void sum_channels_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    long long sN, long long sC, long long sH, long long sW,
    cudaStream_t stream)
{
    long long total_out = (long long)N * H * W;
    int block = 256;
    long long grid_l = (total_out + block - 1) / block;
    int grid = (int)min(grid_l, (long long)2147483647);
    sum_channels_kernel<float><<<grid, block, 0, stream>>>(
        x, y, N, C, H, W, sN, sC, sH, sW, total_out);
}

// -----------------------------------------------------------------------------
// expand_channels_launcher: materializes x.expand(-1, channels_factor, -1, -1)
// Input: x shape [N, 1, H, W], strides [sN, sC, sH, sW] (sC may be arbitrary since size 1, often 0 or H*W)
// Output: y shape [N, channels_factor, H, W], contiguous
// Each y[n,c,h,w] = x[n,0,h,w]
// -----------------------------------------------------------------------------
template <typename T>
__global__ void expand_channels_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int N, int channels_factor, int H, int W,
    long long sN, long long sC, long long sH, long long sW,
    long long total_out) // N*channels_factor*H*W
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < total_out; idx += stride) {
        int w = idx % W;
        int tmp = idx / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % channels_factor;
        int n = tmp / channels_factor;
        // read from x at [n,0,h,w]
        const T* x_ptr = x + (long long)n * sN + (long long)0 * sC + (long long)h * sH + (long long)w * sW;
        y[idx] = *x_ptr;
    }
}

extern "C" void expand_channels_launcher(
    const float* x,
    float* y,
    int N, int channels_factor, int H, int W,
    long long sN, long long sC, long long sH, long long sW,
    cudaStream_t stream)
{
    long long total_out = (long long)N * channels_factor * H * W;
    int block = 256;
    long long grid_l = (total_out + block - 1) / block;
    int grid = (int)min(grid_l, (long long)2147483647);
    expand_channels_kernel<float><<<grid, block, 0, stream>>>(
        x, y, N, channels_factor, H, W, sN, sC, sH, sW, total_out);
}

// -----------------------------------------------------------------------------
// pixel_shuffle_launcher: performs nn.PixelShuffle(r)
// Input: x shape [N, C_in, H, W] (C_in divisible by r*r), strides [sN, sC, sH, sW]
// Output: y shape [N, C_out, H*r, W*r] with C_out = C_in / (r*r)
// Mapping: y[n, c_out, h*r + i, w*r + j] = x[n, c_out * r*r + i*r + j, h, w]
// Write contiguous output.
// -----------------------------------------------------------------------------
template <typename T>
__global__ void pixel_shuffle_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    int N, int C_in, int H, int W, int r,
    long long sN, long long sC, long long sH, long long sW,
    int C_out,
    long long total_out) // N * C_out * H*r * W*r
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < total_out; idx += stride) {
        int ow = idx % (W * r);
        long long tmp = idx / (W * r);
        int oh = tmp % (H * r);
        tmp /= (H * r);
        int c_out = tmp % C_out;
        int n = tmp / C_out;
        int i = oh / r;
        int j = ow / r;
        int h = i;
        int w = j;
        int c_in = c_out * r * r + (oh % r) * r + (ow % r);
        const T* x_ptr = x + (long long)n * sN + (long long)c_in * sC + (long long)h * sH + (long long)w * sW;
        y[idx] = *x_ptr;
    }
}

extern "C" void pixel_shuffle_launcher(
    const float* x,
    float* y,
    int N, int C_in, int H, int W, int r,
    long long sN, long long sC, long long sH, long long sW,
    cudaStream_t stream)
{
    int C_out = C_in / (r * r);
    long long total_out = (long long)N * C_out * H * r * W * r;
    int block = 256;
    long long grid_l = (total_out + block - 1) / block;
    int grid = (int)min(grid_l, (long long)2147483647);
    pixel_shuffle_kernel<float><<<grid, block, 0, stream>>>(
        x, y, N, C_in, H, W, r, sN, sC, sH, sW, C_out, total_out);
}
