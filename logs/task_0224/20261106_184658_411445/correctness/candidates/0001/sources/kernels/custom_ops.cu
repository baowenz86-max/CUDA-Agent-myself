// CUDA kernels for hardtanh and avgpool2d custom ops
#include <cuda_runtime.h>

// Forward declarations not needed since all in same file

// Kernel: hardtanh forward (clamp input to [min_val, max_val])
__global__ void hardtanh_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long total_elements,
    float min_val,
    float max_val)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float val = input[idx];
        val = fminf(fmaxf(val, min_val), max_val);
        output[idx] = val;
    }
}

// Kernel: avgpool2d forward with kernel_size=2, stride=1, padding=0
// Input is assumed contiguous NCHW float32 tensor, output contiguous NCHW
__global__ void avgpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = N * C * H_out * W_out;
    if (idx >= total_out) return;
    
    // Decompose linear index into n,c,oh,ow
    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;
    
    // Input top-left corner for this output
    int ih = oh;  // stride 1, padding 0
    int iw = ow;
    
    // Base pointers for this n,c
    const float* in_base = input + ((long long)n * C + c) * H * W;
    float* out_base = output + ((long long)n * C + c) * H_out * W_out;
    
    // Average the 2x2 window
    // Since kernel_size=2, stride=1, padding=0, the window always fits (except maybe last row/col?)
    // Output dims: H_out = H - 1, W_out = W - 1, and with kernel 2, the window indices ih, ih+1, iw, iw+1 are always within bounds (0..H-1, 0..W-1)
    float sum = 0.0f;
    sum += in_base[ih * W + iw];
    sum += in_base[ih * W + iw + 1];
    sum += in_base[(ih + 1) * W + iw];
    sum += in_base[(ih + 1) * W + iw + 1];
    
    out_base[oh * W_out + ow] = sum * 0.25f;
}

// Launcher: hardtanh_forward
// Input/output are assumed contiguous float32 CUDA tensors of size total_elements
extern "C" void hardtanh_forward_launcher(
    float* output,
    const float* input,
    long long total_elements,
    float min_val,
    float max_val,
    cudaStream_t stream)
{
    const int block_size = 256;
    long long grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 0) {
        hardtanh_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
            output, input, total_elements, min_val, max_val);
    }
}

// Launcher: avgpool2d_forward
// Input/output are assumed contiguous float32 CUDA tensors with shape [N,C,H,W] and [N,C,H_out,W_out]
extern "C" void avgpool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernel_size, int stride, int padding,
    cudaStream_t stream)
{
    // Only support kernel_size=2, stride=1, padding=0 as per contract, but we keep params for generality
    // In this implementation we ignore kernel_size, stride, padding and assume exactly 2x2 stride 1 no padding.
    // Could add checks, but launcher should be simple.
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;
    
    int total_out = N * C * H_out * W_out;
    const int block_size = 256;
    int grid_size = (total_out + block_size - 1) / block_size;
    if (grid_size > 0) {
        avgpool2d_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output, N, C, H, W, H_out, W_out);
    }
}
