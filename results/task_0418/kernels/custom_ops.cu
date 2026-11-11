#include <cuda_runtime.h>
#include <cmath>

// Lp pooling kernel.
// input: pointer to input data (NCHW contiguous assumed, but can be adjusted with strides if needed)
// output: pointer to output data (contiguous)
// norm_type: p value, double
// We specialize for p=2 with a separate kernel for speed and accuracy.
// For general p, use powf and abs.
// The launcher selects kernel based on norm_type == 2.0.

// General p-norm kernel
__global__ void lp_pool2d_generic_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernel_size, int stride,
    double p)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z / C;
    int c = blockIdx.z % C;

    if (ow >= outW || oh >= outH) return;

    int h_start = oh * stride;
    int w_start = ow * stride;

    float sum_p = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h = h_start + kh;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int w = w_start + kw;
            float val = input[((n * C + c) * H + h) * W + w];
            sum_p += powf(fabsf(val), (float)p);
        }
    }
    output[((n * C + c) * outH + oh) * outW + ow] = powf(sum_p, 1.0f / (float)p);
}

// Specialized p=2 kernel (Euclidean norm)
__global__ void lp_pool2d_l2_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernel_size, int stride)
{
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z / C;
    int c = blockIdx.z % C;

    if (ow >= outW || oh >= outH) return;

    int h_start = oh * stride;
    int w_start = ow * stride;

    float sum_sq = 0.0f;
    for (int kh = 0; kh < kernel_size; ++kh) {
        int h = h_start + kh;
        for (int kw = 0; kw < kernel_size; ++kw) {
            int w = w_start + kw;
            float val = input[((n * C + c) * H + h) * W + w];
            sum_sq += val * val;
        }
    }
    output[((n * C + c) * outH + oh) * outW + ow] = sqrtf(sum_sq);
}

// Launcher: allocates output in binding, here we just launch.
// Input pointer assumed contiguous NCHW. We don't handle non-contiguous here; binding may make contiguous.
// norm_type is double; we cast to float for kernel?
__host__ void lp_pool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernel_size, int stride,
    double norm_type,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y, N * C);

    if (norm_type == 2.0) {
        lp_pool2d_l2_kernel<<<grid, block, 0, stream>>>(
            input, output, N, C, H, W, outH, outW, kernel_size, stride);
    } else {
        lp_pool2d_generic_kernel<<<grid, block, 0, stream>>>(
            input, output, N, C, H, W, outH, outW, kernel_size, stride, norm_type);
    }
}

// C-interface launcher for binding (extern "C")
extern "C" void lp_pool2d_launcher_ext(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t outH, int64_t outW,
    int64_t kernel_size, int64_t stride,
    double norm_type,
    cudaStream_t stream)
{
    lp_pool2d_launcher(input, output, (int)N, (int)C, (int)H, (int)W,
                       (int)outH, (int)outW, (int)kernel_size, (int)stride,
                       norm_type, stream);
}
