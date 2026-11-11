#include <cuda_runtime.h>

// CUDA kernel for constant padding (F.pad with mode='constant', value=0)
// Input: x of shape [N, C, H, W] contiguous
template <typename scalar_t>
__global__ void pad_constant_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ in,
    int N, int C, int H_in, int W_in,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int H_out, int W_out)
{
    int total = N * C * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int w = i % W_out;
        int h = (i / W_out) % H_out;
        int c = (i / (W_out * H_out)) % C;
        int n = i / (W_out * H_out * C);

        int in_h = h - pad_top;
        int in_w = w - pad_left;
        if (in_h >= 0 && in_h < H_in && in_w >= 0 && in_w < W_in) {
            out[i] = in[((n * C + c) * H_in + in_h) * W_in + in_w];
        } else {
            out[i] = static_cast<scalar_t>(0);
        }
    }
}

// CUDA kernel for mean along dimension (dim=1) for float32
// Input: x of shape [N, C, H, W] contiguous
// Output: y of shape [N, H, W] (when dim=1)
// Since C is 1 in the current contract, we can copy directly.
// But we implement general reduction with a single thread per output element.
template <typename scalar_t>
__global__ void mean_dim1_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ in,
    int N, int C, int H, int W)
{
    int total = N * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int w = i % W;
        int h = (i / W) % H;
        int n = i / (H * W);

        scalar_t sum = 0;
        for (int c = 0; c < C; ++c) {
            sum += in[((n * C + c) * H + h) * W + w];
        }
        out[i] = sum / static_cast<scalar_t>(C);
    }
}

// Launcher for pad_forward
extern "C" void pad_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    cudaStream_t stream)
{
    int H_out = H + pad_top + pad_bottom;
    int W_out = W + pad_left + pad_right;
    int total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Cap blocks to avoid grid dimension limits, use grid-stride loop inside
    if (blocks > 65535) blocks = 65535;
    pad_constant_kernel<float><<<blocks, threads, 0, stream>>>(
        output, input, N, C, H, W, pad_left, pad_right, pad_top, pad_bottom, H_out, W_out);
}

// Launcher for mean_dim_forward (dim=1 only)
extern "C" void mean_dim_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    cudaStream_t stream)
{
    int total = N * H * W;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    mean_dim1_kernel<float><<<blocks, threads, 0, stream>>>(
        output, input, N, C, H, W);
}
