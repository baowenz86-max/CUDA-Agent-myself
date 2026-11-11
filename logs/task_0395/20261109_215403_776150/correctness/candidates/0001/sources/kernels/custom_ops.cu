#include <cuda_runtime.h>
#include <cstdint>

// Average Pooling 2D forward kernel
// input: (N, C, H, W) with strides in elements (not bytes), layout supports strides
// output: (N, C, outH, outW) contiguous
// kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w
// count_include_pad = true (divisor is kernel_h*kernel_w)
__global__ void avg_pool2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t outH, int64_t outW,
    int64_t inputNStride, int64_t inputCStride, int64_t inputHStride, int64_t inputWStride,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t padding_h, int64_t padding_w)
{
    int64_t total = N * C * outH * outW;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += gridDim.x * blockDim.x) {
        int64_t w_out = idx % outW;
        int64_t h_out = (idx / outW) % outH;
        int64_t c = (idx / (outW * outH)) % C;
        int64_t n = idx / (outW * outH * C);

        float sum = 0.0f;
        int64_t h_start = h_out * stride_h - padding_h;
        int64_t w_start = w_out * stride_w - padding_w;
        int64_t h_end = h_start + kernel_h;
        int64_t w_end = w_start + kernel_w;

        const float* base = input + n * inputNStride + c * inputCStride;
        for (int64_t kh = h_start; kh < h_end; ++kh) {
            for (int64_t kw = w_start; kw < w_end; ++kw) {
                if (kh >= 0 && kh < H && kw >= 0 && kw < W) {
                    sum += base[kh * inputHStride + kw * inputWStride];
                }
                // else treat as zero padding; divisor includes it
            }
        }
        output[idx] = sum / (float)(kernel_h * kernel_w);
    }
}

// LeakyReLU forward kernel (elementwise)
__global__ void leaky_relu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total,
    float negative_slope)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += gridDim.x * blockDim.x) {
        float val = input[idx];
        output[idx] = (val >= 0.0f) ? val : negative_slope * val;
    }
}

extern "C" void avg_pool2d_forward(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t inputNStride, int64_t inputCStride, int64_t inputHStride, int64_t inputWStride,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t padding_h, int64_t padding_w,
    bool ceil_mode,
    cudaStream_t stream)
{
    // Compute output dimensions
    int64_t outH, outW;
    if (ceil_mode) {
        outH = (H + 2 * padding_h - kernel_h + stride_h - 1) / stride_h + 1;
        outW = (W + 2 * padding_w - kernel_w + stride_w - 1) / stride_w + 1;
    } else {
        outH = (H + 2 * padding_h - kernel_h) / stride_h + 1;
        outW = (W + 2 * padding_w - kernel_w) / stride_w + 1;
    }

    int64_t total = N * C * outH * outW;
    if (total <= 0) return;

    const int block = 256;
    int grid = (int)((total + block - 1) / block);
    if (grid > 65535) grid = 65535; // grid-stride loop handles rest

    avg_pool2d_forward_kernel<<<grid, block, 0, stream>>>(
        input, output,
        N, C, H, W, outH, outW,
        inputNStride, inputCStride, inputHStride, inputWStride,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w);
}

extern "C" void leaky_relu_forward(
    const float* input,
    float* output,
    int64_t total,
    double negative_slope,
    cudaStream_t stream)
{
    if (total <= 0) return;
    float slope = (float)negative_slope;
    const int block = 256;
    int grid = (int)((total + block - 1) / block);
    if (grid > 65535) grid = 65535;
    leaky_relu_forward_kernel<<<grid, block, 0, stream>>>(
        input, output, total, slope);
}
