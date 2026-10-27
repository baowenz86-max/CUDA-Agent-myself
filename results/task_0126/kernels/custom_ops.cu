// custom_ops.cu - CUDA kernels for optimized model
#include <cuda_runtime.h>
#include <cmath>

// Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
__device__ __forceinline__ float gelu_exact(float x) {
    const float sqrt2 = 1.4142135623730951f;
    return 0.5f * x * (1.0f + erff(x / sqrt2));
}

// GELU kernel over 3D tensor with arbitrary strides.
// output assumed contiguous with same logical shape as input.
__global__ void gelu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int numel,
                            int C,
                            int L,
                            int stride_b,
                            int stride_c,
                            int stride_l) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // Decode linear index into (b, c, l) using logical dimensions
    int b = idx / (C * L);
    int rem = idx - b * (C * L);
    int c = rem / L;
    int l = rem - c * L;

    // Compute input offset using strides
    int input_index = b * stride_b + c * stride_c + l * stride_l;
    output[idx] = gelu_exact(input[input_index]);
}

// 1D max pooling kernel over last dimension for each batch/channel.
// output contiguous shape [B, C, out_len].
__global__ void maxpool1d_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int B,
                                 int C,
                                 int L,
                                 int out_len,
                                 int kernel_size,
                                 int stride,
                                 int padding,
                                 int dilation,
                                 int stride_b,
                                 int stride_c,
                                 int stride_l) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = B * C * out_len;
    if (idx >= total_out) return;

    // decompose idx into (b, c, ol)
    int b = idx / (C * out_len);
    int rem = idx - b * (C * out_len);
    int c = rem / out_len;
    int ol = rem - c * out_len;

    float max_val = -INFINITY;
    // For each kernel element
    for (int k = 0; k < kernel_size; ++k) {
        int il = ol * stride + k * dilation - padding;
        if (il >= 0 && il < L) {
            int input_index = b * stride_b + c * stride_c + il * stride_l;
            float val = input[input_index];
            if (val > max_val) max_val = val;
        }
    }
    output[idx] = max_val;
}

// Launchers
extern "C" void gelu_forward_launcher(const float* input, float* output, int numel, int C, int L, int stride_b, int stride_c, int stride_l, cudaStream_t stream) {
    int block = 256;
    int grid = (numel + block - 1) / block;
    gelu_kernel<<<grid, block, 0, stream>>>(input, output, numel, C, L, stride_b, stride_c, stride_l);
}

extern "C" void maxpool1d_forward_launcher(const float* input, float* output, int B, int C, int L, int out_len, int kernel_size, int stride, int padding, int dilation, int stride_b, int stride_c, int stride_l, cudaStream_t stream) {
    int block = 256;
    int total_out = B * C * out_len;
    int grid = (total_out + block - 1) / block;
    maxpool1d_kernel<<<grid, block, 0, stream>>>(input, output, B, C, L, out_len, kernel_size, stride, padding, dilation, stride_b, stride_c, stride_l);
}
