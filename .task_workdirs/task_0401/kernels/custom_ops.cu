#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

// -----------------------------------------------------------------------------
// custom_max_dim
// 沿 dim=1 (C 维) 求最大值，keepdim=True
// 输入: x float32 [B, C, H, W]，支持任意 stride
// 输出: values float32 [B, 1, H, W]，indices int64 [B, 1, H, W]
// 相等元素返回较小索引
// -----------------------------------------------------------------------------

__global__ void max_dim_kernel(const float* __restrict__ input,
                               float* __restrict__ values,
                               int64_t* __restrict__ indices,
                               int B, int C, int H, int W,
                               long long input_stride_b,
                               long long input_stride_c,
                               long long input_stride_h,
                               long long input_stride_w,
                               long long output_stride_b,
                               long long output_stride_h,
                               long long output_stride_w) {
    long long total = (long long)B * H * W;
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        int b = idx / ((long long)H * W);
        int rem = idx % ((long long)H * W);
        int h = rem / W;
        int w = rem % W;

        const float* base = input + b * input_stride_b + h * input_stride_h + w * input_stride_w;
        float max_val = -FLT_MAX;
        int max_idx = -1;
        for (int c = 0; c < C; ++c) {
            float val = base[c * input_stride_c];
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
            // 若相等，不更新，保留较小索引
        }
        // 输出位置 [b, 0, h, w]
        long long out_offset = b * output_stride_b + h * output_stride_h + w * output_stride_w;
        values[out_offset] = max_val;
        indices[out_offset] = (int64_t)max_idx;
    }
}

extern "C" void custom_max_dim_launcher(const float* input,
                                        float* values,
                                        int64_t* indices,
                                        int B, int C, int H, int W,
                                        long long input_stride_b,
                                        long long input_stride_c,
                                        long long input_stride_h,
                                        long long input_stride_w,
                                        long long output_stride_b,
                                        long long output_stride_h,
                                        long long output_stride_w,
                                        cudaStream_t stream) {
    long long total = (long long)B * H * W;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid limit safe
    max_dim_kernel<<<blocks, threads, 0, stream>>>(
        input, values, indices, B, C, H, W,
        input_stride_b, input_stride_c, input_stride_h, input_stride_w,
        output_stride_b, output_stride_h, output_stride_w);
    cudaGetLastError();
}

// -----------------------------------------------------------------------------
// custom_max_pool2d
// 2D max pooling kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=false
// 输入: x float32 [B, 1, H, W]，支持任意 stride
// 输出: y float32 [B, 1, H_out, W_out]
// 负无穷填充 padding 区域
// H_out = floor((H + 2*padding - dilation*(kernel_size-1) - 1)/stride + 1)
// -----------------------------------------------------------------------------

__global__ void max_pool2d_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int B, int H, int W,
                                  int H_out, int W_out,
                                  int kernel_size, int stride, int padding, int dilation,
                                  long long input_stride_b,
                                  long long input_stride_h,
                                  long long input_stride_w,
                                  long long output_stride_b,
                                  long long output_stride_h,
                                  long long output_stride_w) {
    long long total = (long long)B * H_out * W_out;
    for (long long idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        int b = idx / ((long long)H_out * W_out);
        int rem = idx % ((long long)H_out * W_out);
        int oh = rem / W_out;
        int ow = rem % W_out;

        float max_val = -FLT_MAX;
        for (int kh = 0; kh < kernel_size; ++kh) {
            int ih = oh * stride - padding + kh * dilation;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < kernel_size; ++kw) {
                int iw = ow * stride - padding + kw * dilation;
                if (iw < 0 || iw >= W) continue;
                float val = input[b * input_stride_b + ih * input_stride_h + iw * input_stride_w];
                if (val > max_val) max_val = val;
            }
        }
        long long out_offset = b * output_stride_b + oh * output_stride_h + ow * output_stride_w;
        output[out_offset] = max_val;
    }
}

extern "C" void custom_max_pool2d_launcher(const float* input,
                                           float* output,
                                           int B, int H, int W,
                                           int H_out, int W_out,
                                           int kernel_size, int stride, int padding, int dilation,
                                           long long input_stride_b,
                                           long long input_stride_h,
                                           long long input_stride_w,
                                           long long output_stride_b,
                                           long long output_stride_h,
                                           long long output_stride_w,
                                           cudaStream_t stream) {
    long long total = (long long)B * H_out * W_out;
    if (total <= 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    max_pool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, B, H, W, H_out, W_out,
        kernel_size, stride, padding, dilation,
        input_stride_b, input_stride_h, input_stride_w,
        output_stride_b, output_stride_h, output_stride_w);
    cudaGetLastError();
}
