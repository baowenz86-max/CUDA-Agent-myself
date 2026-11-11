#include <cuda_runtime.h>

// ReLU kernel: out[i] = max(in[i], 0)
// Assumes contiguous 1D indexing over numel elements.
__global__ void relu_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
    }
}

// AvgPool2d kernel: output[n,c,oh,ow] = sum over kh,kw of input[n,c,oh*stride+kh, ow*stride+kw] / (kernel_size*kernel_size)
// Input must be contiguous NCHW float32.
__global__ void avgpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int kernel_size, int stride,
    int64_t outH, int64_t outW
) {
    int64_t total_out = N * C * outH * outW;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t step = blockDim.x * gridDim.x;
    float inv_area = 1.0f / (float)(kernel_size * kernel_size);

    for (int64_t i = idx; i < total_out; i += step) {
        int64_t ow = i % outW;
        int64_t oh = (i / outW) % outH;
        int64_t c = (i / (outW * outH)) % C;
        int64_t n = i / (outW * outH * C);

        float sum = 0.0f;
        for (int kh = 0; kh < kernel_size; ++kh) {
            int64_t ih = oh * stride + kh;
            if (ih >= H) break; // padding=0, ignore out-of-bounds; PyTorch AvgPool2d ignores padding elements
            for (int kw = 0; kw < kernel_size; ++kw) {
                int64_t iw = ow * stride + kw;
                if (iw < W) {
                    int64_t input_idx = ((n * C + c) * H + ih) * W + iw;
                    sum += input[input_idx];
                }
            }
        }
        output[i] = sum * inv_area;
    }
}

// Launcher for ReLU
extern "C" void relu_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream
) {
    if (numel <= 0) return;
    const int block = 256;
    int64_t grid64 = (numel + block - 1) / block;
    int grid = (int)min(grid64, (int64_t)65535); // cap grid size; kernel uses stride loop for larger
    relu_kernel<<<grid, block, 0, stream>>>(input, output, numel);
}

// Launcher for AvgPool2d
extern "C" void avgpool2d_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int kernel_size, int stride,
    cudaStream_t stream
) {
    int64_t outH = (H - kernel_size) / stride + 1;
    int64_t outW = (W - kernel_size) / stride + 1;
    int64_t total_out = N * C * outH * outW;
    if (total_out <= 0) return;

    const int block = 256;
    int64_t grid64 = (total_out + block - 1) / block;
    int grid = (int)min(grid64, (int64_t)65535);
    avgpool2d_kernel<<<grid, block, 0, stream>>>(
        input, output,
        N, C, H, W,
        kernel_size, stride,
        outH, outW
    );
}
