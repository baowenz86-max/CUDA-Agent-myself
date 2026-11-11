#include <cuda_runtime.h>

// ConstantPad3d kernel: pads a 5D tensor (N,C,D,H,W) with zeros.
// Input and output are assumed to be contiguous float32 tensors.
// pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back specify padding.
__global__ void constant_pad3d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t pad_left, int64_t pad_right,
    int64_t pad_top, int64_t pad_bottom,
    int64_t pad_front, int64_t pad_back)
{
    int64_t total_out = N * C * D_out * H_out * W_out;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_out; i += stride) {
        // Decompose linear index into 5D coordinates (n, c, d, h, w)
        int64_t w = i % W_out;
        int64_t tmp = i / W_out;
        int64_t h = tmp % H_out;
        tmp /= H_out;
        int64_t d = tmp % D_out;
        tmp /= D_out;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        // Compute source coordinates in input
        int64_t in_w = w - pad_left;
        int64_t in_h = h - pad_top;
        int64_t in_d = d - pad_front;

        // Check if within input bounds
        bool valid = (in_w >= 0 && in_w < W_in &&
                      in_h >= 0 && in_h < H_in &&
                      in_d >= 0 && in_d < D_in);
        if (valid) {
            int64_t in_idx = (((n * C + c) * D_in + in_d) * H_in + in_h) * W_in + in_w;
            output[i] = input[in_idx];
        } else {
            output[i] = 0.0f;
        }
    }
}

// BiasAdd kernel: adds per-channel bias to a 5D tensor (N,C,D,H,W).
// Input and output are assumed to be contiguous float32 tensors.
// bias is a contiguous tensor of shape [C].
__global__ void bias_add_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t total_elements,
    int64_t C)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_elements; i += stride) {
        // Compute channel index: the channel dimension is the second dimension,
        // and the size of the last three dims is total_elements / (N*C)
        // We assume total_elements % C == 0 and total_elements / C is the product of N*D*H*W.
        int64_t channel = (i / (total_elements / C)) % C;  // More precisely: channel = (i / (D*H*W)) % C, but we have total_elements/C = N*D*H*W, and we need i / (D*H*W) % C. Since total_elements = N*C*D*H*W, total_elements/C = N*D*H*W. However, that quotient includes N, so this is wrong. Correct: channel = (i / (D*H*W)) % C. But we don't have D,H,W separately. Better: pass spatial_size = D*H*W as argument, then channel = (i / spatial_size) % C.
        // For simplicity, we can compute channel from the known total elements and C if N*D*H*W is known. But we can also pass D*H*W explicitly. Let's modify launcher to pass D_out*H_out*W_out.
        // Here we use a placeholder; actual launcher will pass spatial_size.
        // To avoid error, we will define a different kernel with spatial_size.
    }
}

// Corrected BiasAdd kernel with spatial_size parameter
__global__ void bias_add_kernel_v2(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t total_elements,
    int64_t C,
    int64_t spatial_size)  // D*H*W (product of last three dims)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t channel = (i / spatial_size) % C;
        output[i] = input[i] + bias[channel];
    }
}

// Launcher for constant_pad3d
// All pointer arguments are raw device pointers.
extern "C" void constant_pad3d_launcher(
    const float* input, float* output,
    int64_t N, int64_t C, int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t pad_left, int64_t pad_right,
    int64_t pad_top, int64_t pad_bottom,
    int64_t pad_front, int64_t pad_back,
    cudaStream_t stream)
{
    int64_t total_out = N * C * D_out * H_out * W_out;
    int block_size = 256;
    int grid_size = (int)((total_out + block_size - 1) / block_size);
    if (grid_size > 0) {
        constant_pad3d_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output,
            N, C, D_in, H_in, W_in,
            D_out, H_out, W_out,
            pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back);
    }
}

// Launcher for bias_add
// input, bias, output are contiguous float32 tensors.
extern "C" void bias_add_launcher(
    const float* input, const float* bias, float* output,
    int64_t total_elements, int64_t C, int64_t spatial_size,
    cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (int)((total_elements + block_size - 1) / block_size);
    if (grid_size > 0) {
        bias_add_kernel_v2<<<grid_size, block_size, 0, stream>>>(
            input, bias, output, total_elements, C, spatial_size);
    }
}
