// CUDA kernels for custom_abs and custom_avg_pool2d
// All kernels and launchers are placed in this single .cu file.
// No torch/ATen/c10/pybind dependencies; only raw pointers, scalars and cudaStream_t.

#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// custom_abs: element-wise absolute value for float32 CUDA tensors.
// Supports arbitrary strides via input pointer arithmetic.
// -----------------------------------------------------------------------------
__global__ void custom_abs_kernel(float* __restrict__ output,
                                  const float* __restrict__ input,
                                  int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = fabsf(input[i]);
    }
}

extern "C" void custom_abs_launcher(float* output,
                                    const float* input,
                                    int64_t numel,
                                    cudaStream_t stream) {
    if (numel == 0) return;
    const int threads = 256;
    int64_t blocks64 = (numel + threads - 1) / threads;
    if (blocks64 > 2147483647LL) blocks64 = 2147483647LL; // grid dimension limit
    int blocks = (int)blocks64;
    custom_abs_kernel<<<blocks, threads, 0, stream>>>(output, input, numel);
}

// -----------------------------------------------------------------------------
// custom_avg_pool2d: average pooling with kernel_size=(3,5), stride=1,
// padding=(1,2). Output has the same spatial dimensions as input.
// Input is expected to be float32 CUDA tensor with arbitrary strides.
// PyTorch AvgPool2d default count_include_pad=True, so division is by
// kernel_size product (3*5=15), regardless of padding.
// -----------------------------------------------------------------------------

__global__ void custom_avg_pool2d_kernel(float* __restrict__ output,
                                         const float* __restrict__ input,
                                         int64_t N,   // batch
                                         int64_t C,   // channels
                                         int64_t H,   // input height
                                         int64_t W,   // input width
                                         int64_t in_stride_n,
                                         int64_t in_stride_c,
                                         int64_t in_stride_h,
                                         int64_t in_stride_w,
                                         int64_t out_stride_n,
                                         int64_t out_stride_c,
                                         int64_t out_stride_h,
                                         int64_t out_stride_w) {
    // Each thread computes one output element (n, c, oh, ow).
    int64_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    int64_t total = N * C * H * W; // output spatial size = input spatial size
    for (int64_t idx = linear; idx < total; idx += stride) {
        int64_t ow = idx % W;
        int64_t tmp = idx / W;
        int64_t oh = tmp % H;
        tmp /= H;
        int64_t c = tmp % C;
        int64_t n = tmp / C;

        float sum = 0.0f;
        // kernel size: kh=3, kw=5, padding top=1, bottom=1, left=2, right=2
        // PyTorch AvgPool2d with padding=(1,2): pad_h=1, pad_w=2.
        // The same padding is added on both sides symmetric.
        // For each input position (ih, iw) within window:
        // ih_start = oh*stride - pad_h = oh - 1
        // iw_start = ow*stride - pad_w = ow - 2
        for (int kh = 0; kh < 3; ++kh) {
            int64_t ih = (int64_t)oh - 1 + kh; // because pad_h=1
            if (ih < 0 || ih >= H) continue;   // zero padding
            for (int kw = 0; kw < 5; ++kw) {
                int64_t iw = (int64_t)ow - 2 + kw; // because pad_w=2
                if (iw < 0 || iw >= W) continue;   // zero padding
                // Read input value using strides
                const float* in_ptr = input + n * in_stride_n
                                     + c * in_stride_c
                                     + ih * in_stride_h
                                     + iw * in_stride_w;
                sum += *in_ptr;
            }
        }
        // AvgPool2d divides by kernel_size (15) because count_include_pad=True.
        // The zero padding elements do not contribute to sum, but are included
        // in the divisor, exactly matching PyTorch default behavior.
        float avg = sum / 15.0f;
        float* out_ptr = output + n * out_stride_n
                         + c * out_stride_c
                         + oh * out_stride_h
                         + ow * out_stride_w;
        *out_ptr = avg;
    }
}

extern "C" void custom_avg_pool2d_launcher(float* output,
                                           const float* input,
                                           int64_t N,
                                           int64_t C,
                                           int64_t H,
                                           int64_t W,
                                           int64_t in_stride_n,
                                           int64_t in_stride_c,
                                           int64_t in_stride_h,
                                           int64_t in_stride_w,
                                           int64_t out_stride_n,
                                           int64_t out_stride_c,
                                           int64_t out_stride_h,
                                           int64_t out_stride_w,
                                           cudaStream_t stream) {
    int64_t total = N * C * H * W;
    if (total == 0) return;
    const int threads = 256;
    int64_t blocks64 = (total + threads - 1) / threads;
    if (blocks64 > 2147483647LL) blocks64 = 2147483647LL;
    int blocks = (int)blocks64;
    custom_avg_pool2d_kernel<<<blocks, threads, 0, stream>>>(
        output, input, N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w);
}
