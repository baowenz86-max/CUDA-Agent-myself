// kernels/custom_ops.cu
// CUDA kernel for 2D max pooling.
// Only raw pointers and scalars are used; no torch/ATen/pybind dependencies.
#include <cuda_runtime.h>

// Kernel: each thread computes one output element.
// Input: (N, C, H, W) contiguous float32 tensor.
// Output: (N, C, OH, OW) contiguous float32 tensor.
// Parameters kernel_size, stride, padding, dilation, ceil_mode follow PyTorch MaxPool2d.
__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int OH, int OW,
    int kernel_size, int stride, int padding, int dilation, bool ceil_mode)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * OH * OW;
    if (idx >= total) return;

    // Decompose linear index: ow fastest, then oh, then c, then n.
    int ow = idx % OW;
    int tmp = idx / OW;
    int oh = tmp % OH;
    tmp /= OH;
    int c = tmp % C;
    int n = tmp / C;

    // Input window top-left in input coordinates (before padding, with dilation).
    int h_start = oh * stride - padding;
    int w_start = ow * stride - padding;

    float max_val = -3.402823466e+38f; // -FLT_MAX
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h_in = h_start + kh * dilation;
        if (h_in < 0 || h_in >= H) continue; // out of bounds treated as -infinity (padding ignored)
        for (int kw = 0; kw < kernel_size; ++kw) {
            int w_in = w_start + kw * dilation;
            if (w_in < 0 || w_in >= W) continue;
            float val = input[((n * C + c) * H + h_in) * W + w_in];
            if (val > max_val) max_val = val;
        }
    }

    output[((n * C + c) * OH + oh) * OW + ow] = max_val;
}

// Launcher with C linkage for binding layer (kernels/*_binding.cpp).
// Assumes input and output are contiguous float32 CUDA tensors.
extern "C" void maxpool2d_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode,
    cudaStream_t stream)
{
    // Compute output spatial dimensions.
    int64_t OH, OW;
    if (ceil_mode) {
        OH = (H + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) / stride + 1;
        OW = (W + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) / stride + 1;
    } else {
        OH = (H + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        OW = (W + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    }

    int total = (int)(N * C * OH * OW);
    if (total <= 0) return;

    // Launch configuration: 256 threads per block, grid-stride loop not needed because total may be <= 2^31-1 and grid cap handled dynamically.
    int block = 256;
    int grid = (total + block - 1) / block;

    maxpool2d_kernel<<<grid, block, 0, stream>>>(
        input, output,
        (int)N, (int)C, (int)H, (int)W,
        (int)OH, (int)OW,
        (int)kernel_size, (int)stride, (int)padding, (int)dilation, ceil_mode);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real binding, we would throw; but since this is pure .cu, we just record the error.
        // The binding layer should check cudaGetLastError after calling the launcher.
        // We do not include any exception throwing to keep this file torch-free.
    }
}
