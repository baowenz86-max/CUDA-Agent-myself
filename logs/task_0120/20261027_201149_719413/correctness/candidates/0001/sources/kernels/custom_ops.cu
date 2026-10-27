#include <cuda_runtime.h>

// CUDA kernel for torch.nn.functional.unfold with stride=1
// Input: (N, C, H, W) float32 contiguous
// Output: (N, C * kernel_h * kernel_w, L) float32 contiguous
// L = out_h * out_w, where out_h = H + 2*padding_h - dilation_h*(kernel_h-1)
// out_w = W + 2*padding_w - dilation_w*(kernel_w-1)
// Im2col ordering: for each n, for each c, for each kh, for each kw, then for each output spatial location
// Output index = ((n * C + c) * kernel_h + kh) * kernel_w + kw) * L + (oh * out_w + ow)
// Input index = ((n * C + c) * H + (oh + kh*dilation_h - padding_h)) * W + (ow + kw*dilation_w - padding_w)
__global__ void unfold_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int dilation_h, int dilation_w,
    int padding_h, int padding_w,
    int out_h, int out_w, int L)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * kernel_h * kernel_w * L;
    if (idx >= total) return;

    // Decompose idx into (n, c, kh, kw, oh, ow)
    int ow = idx % out_w;
    int tmp = idx / out_w;
    int oh = tmp % out_h;
    tmp /= out_h;
    int kw = tmp % kernel_w;
    tmp /= kernel_w;
    int kh = tmp % kernel_h;
    tmp /= kernel_h;
    int c = tmp % C;
    tmp /= C;
    int n = tmp;

    int ih = oh + kh * dilation_h - padding_h;
    int iw = ow + kw * dilation_w - padding_w;
    float val = 0.0f;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
        val = input[((n * C + c) * H + ih) * W + iw];
    }
    output[idx] = val;
}

// CUDA kernel for elementwise sign (matching torch.sign for float32)
__global__ void sign_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long numel)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < numel) {
        float v = input[idx];
        // torch.sign returns 1.0 for positive, -1.0 for negative, 0.0 for zero, NaN for NaN
        // Use comparisons and copysign to handle all cases including NaN
        output[idx] = (v > 0.0f) ? 1.0f : ((v < 0.0f) ? -1.0f : copysignf(0.0f, v));
    }
}

// Launcher for unfold_forward
// All parameters are raw; the binding layer will allocate output and pass pointers.
extern "C" void unfold_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int dilation_h, int dilation_w,
    int padding_h, int padding_w,
    cudaStream_t stream)
{
    int out_h = H + 2 * padding_h - dilation_h * (kernel_h - 1);
    int out_w = W + 2 * padding_w - dilation_w * (kernel_w - 1);
    int L = out_h * out_w;
    int total = N * C * kernel_h * kernel_w * L;

    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    unfold_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output,
        N, C, H, W,
        kernel_h, kernel_w,
        dilation_h, dilation_w,
        padding_h, padding_w,
        out_h, out_w, L);
}

// Launcher for sign_forward
extern "C" void sign_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream)
{
    const int block_size = 256;
    long long grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 0) {
        sign_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
            input, output, numel);
    }
}
