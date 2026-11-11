#include <cuda_runtime.h>

// Kernel for 3D average pooling with kernel_size=2, stride=2
// input: [N, C, D, H, W], output: [N, C, D/2, H/2, W/2]
// This kernel assumes kernel_size=2, stride=2 for all dimensions.
// It supports arbitrary strides via pointer arithmetic.
__global__ void avg_pool3d_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int D, int H, int W,
    int outD, int outH, int outW,
    long long in_stride_N, long long in_stride_C,
    long long in_stride_D, long long in_stride_H, long long in_stride_W,
    long long out_stride_N, long long out_stride_C,
    long long out_stride_D, long long out_stride_H, long long out_stride_W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * outD * outH * outW;
    int stride = blockDim.x * gridDim.x;

    for (int linear = idx; linear < total; linear += stride) {
        // decode output indices
        int ow = linear % outW;
        int oh = (linear / outW) % outH;
        int od = (linear / (outW * outH)) % outD;
        int c  = (linear / (outW * outH * outD)) % C;
        int n  = linear / (outW * outH * outD * C);

        // input base coordinates (top-left-front corner of pooling cube)
        int iw = ow * 2;
        int ih = oh * 2;
        int id = od * 2;

        const float* in_ptr = input + n * in_stride_N + c * in_stride_C
                              + id * in_stride_D + ih * in_stride_H + iw * in_stride_W;

        float sum = 0.0f;
        // 2x2x2 averaging, weights are 1/8 each
        #pragma unroll
        for (int dz = 0; dz < 2; ++dz) {
            #pragma unroll
            for (int dy = 0; dy < 2; ++dy) {
                #pragma unroll
                for (int dx = 0; dx < 2; ++dx) {
                    sum += in_ptr[dz * in_stride_D + dy * in_stride_H + dx * in_stride_W];
                }
            }
        }

        float* out_ptr = output + n * out_stride_N + c * out_stride_C
                         + od * out_stride_D + oh * out_stride_H + ow * out_stride_W;
        *out_ptr = sum * 0.125f;
    }
}

// Kernel for sum over channel dimension (dim=1)
// input: [N, C, D, H, W], output: [N, D, H, W]
// Supports C >= 1, using a loop over C (though for this task C=1)
__global__ void sum_dim1_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int N, int C, int D, int H, int W,
    long long in_stride_N, long long in_stride_C,
    long long in_stride_D, long long in_stride_H, long long in_stride_W,
    long long out_stride_N, long long out_stride_D,
    long long out_stride_H, long long out_stride_W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * D * H * W;
    int stride = blockDim.x * gridDim.x;

    for (int linear = idx; linear < total; linear += stride) {
        int w = linear % W;
        int h = (linear / W) % H;
        int d = (linear / (W * H)) % D;
        int n = linear / (W * H * D);

        const float* in_ptr = input + n * in_stride_N
                              + d * in_stride_D + h * in_stride_H + w * in_stride_W;
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            sum += in_ptr[c * in_stride_C];
        }

        float* out_ptr = output + n * out_stride_N + d * out_stride_D
                         + h * out_stride_H + w * out_stride_W;
        *out_ptr = sum;
    }
}

extern "C" void avg_pool3d_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int D, int H, int W,
    int outD, int outH, int outW,
    long long in_stride_N, long long in_stride_C,
    long long in_stride_D, long long in_stride_H, long long in_stride_W,
    long long out_stride_N, long long out_stride_C,
    long long out_stride_D, long long out_stride_H, long long out_stride_W,
    cudaStream_t stream)
{
    int total = N * C * outD * outH * outW;
    int block_size = 256;
    // use enough blocks to cover total with grid-stride loop
    int grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535; // max grid size in x dimension
    if (grid_size == 0) return;

    avg_pool3d_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input,
        N, C, D, H, W,
        outD, outH, outW,
        in_stride_N, in_stride_C, in_stride_D, in_stride_H, in_stride_W,
        out_stride_N, out_stride_C, out_stride_D, out_stride_H, out_stride_W);
}

extern "C" void sum_dim1_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int D, int H, int W,
    long long in_stride_N, long long in_stride_C,
    long long in_stride_D, long long in_stride_H, long long in_stride_W,
    long long out_stride_N, long long out_stride_D,
    long long out_stride_H, long long out_stride_W,
    cudaStream_t stream)
{
    int total = N * D * H * W;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;
    if (grid_size == 0) return;

    sum_dim1_kernel<<<grid_size, block_size, 0, stream>>>(
        output, input,
        N, C, D, H, W,
        in_stride_N, in_stride_C, in_stride_D, in_stride_H, in_stride_W,
        out_stride_N, out_stride_D, out_stride_H, out_stride_W);
}
