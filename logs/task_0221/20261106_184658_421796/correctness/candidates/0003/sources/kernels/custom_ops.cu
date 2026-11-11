#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cfloat>

// -----------------------------------------------------------------------------
// custom_hardtanh
// -----------------------------------------------------------------------------
__global__ void hardtanh_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int64_t numel,
                                float min_val,
                                float max_val) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; idx < numel; idx += stride) {
        float v = input[idx];
        v = fminf(fmaxf(v, min_val), max_val);
        output[idx] = v;
    }
}

extern "C" void custom_hardtanh_launcher(const float* input,
                                          float* output,
                                          int64_t numel,
                                          float min_val,
                                          float max_val,
                                          cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) {
        grid_size = 2147483647LL;
    }
    hardtanh_kernel<<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(
        input, output, numel, min_val, max_val);
}

// -----------------------------------------------------------------------------
// custom_max_pool2d
// -----------------------------------------------------------------------------
__global__ void max_pool2d_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int batch_size,
                                  int channels,
                                  int input_height,
                                  int input_width,
                                  int output_height,
                                  int output_width,
                                  int kernel_h,
                                  int kernel_w,
                                  int stride_h,
                                  int stride_w,
                                  int padding_h,
                                  int padding_w,
                                  int dilation_h,
                                  int dilation_w) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * output_height * output_width;
    if (out_idx >= total_outputs) return;

    // 分解输出索引为 (n, c, oh, ow)
    int ow = out_idx % output_width;
    int tmp = out_idx / output_width;
    int oh = tmp % output_height;
    tmp /= output_height;
    int c = tmp % channels;
    int n = tmp / channels;

    // 计算输入窗口起始位置
    int h_start = oh * stride_h - padding_h;
    int w_start = ow * stride_w - padding_w;

    float max_val = -FLT_MAX;
    for (int kh = 0; kh < kernel_h; ++kh) {
        int ih = h_start + kh * dilation_h;
        if (ih < 0 || ih >= input_height) continue;
        for (int kw = 0; kw < kernel_w; ++kw) {
            int iw = w_start + kw * dilation_w;
            if (iw < 0 || iw >= input_width) continue;
            // 输入张量形状 [N, C, H, W]，连续存储
            int64_t input_idx = ((static_cast<int64_t>(n) * channels + c) * input_height + ih) * input_width + iw;
            float v = input[input_idx];
            max_val = fmaxf(max_val, v);
        }
    }
    output[out_idx] = max_val;
}

extern "C" void custom_max_pool2d_launcher(const float* input,
                                           float* output,
                                           int batch_size,
                                           int channels,
                                           int input_height,
                                           int input_width,
                                           int output_height,
                                           int output_width,
                                           int kernel_h,
                                           int kernel_w,
                                           int stride_h,
                                           int stride_w,
                                           int padding_h,
                                           int padding_w,
                                           int dilation_h,
                                           int dilation_w,
                                           cudaStream_t stream) {
    int total_outputs = batch_size * channels * output_height * output_width;
    const int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;
    max_pool2d_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, batch_size, channels, input_height, input_width,
        output_height, output_width, kernel_h, kernel_w, stride_h, stride_w,
        padding_h, padding_w, dilation_h, dilation_w);
}

// -----------------------------------------------------------------------------
// custom_flatten
// -----------------------------------------------------------------------------
// 输入 [N, C, H, W] 且连续，输出 [N, C*H*W] 连续。
// 由于模型中新版 max_pool 输出为连续，输入通常连续，但为安全仍使用线性拷贝。
__global__ void flatten_copy_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; idx < numel; idx += stride) {
        output[idx] = input[idx];
    }
}

extern "C" void custom_flatten_launcher(const float* input,
                                        float* output,
                                        int64_t numel,
                                        cudaStream_t stream) {
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    if (grid_size > 2147483647LL) {
        grid_size = 2147483647LL;
    }
    flatten_copy_kernel<<<static_cast<unsigned int>(grid_size), block_size, 0, stream>>>(
        input, output, numel);
}

// -----------------------------------------------------------------------------
// custom_linear (cuBLAS GEMM)
// -----------------------------------------------------------------------------
// 前置声明 add_bias_kernel，避免在 launcher 中调用前未定义
__global__ void add_bias_kernel(float* __restrict__ output,
                                const float* __restrict__ bias,
                                int N,
                                int out_features);

extern "C" void custom_linear_launcher(const float* input,   // [N, in_features]
                                       const float* weight,  // [out_features, in_features]
                                       const float* bias,    // [out_features] 或 nullptr
                                       float* output,        // [N, out_features]
                                       int N,
                                       int in_features,
                                       int out_features,
                                       cudaStream_t stream) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(handle,
                                        CUBLAS_OP_N, CUBLAS_OP_T,
                                        N, out_features, in_features,
                                        &alpha,
                                        input, in_features,
                                        weight, in_features,
                                        &beta,
                                        output, N);

    if (status != CUBLAS_STATUS_SUCCESS) {
        // 简单的错误处理：不抛出，依赖后续 error check
    }

    if (bias != nullptr) {
        // 添加 bias
        int block_size = 256;
        int grid_size = (N * out_features + block_size - 1) / block_size;
        add_bias_kernel<<<grid_size, block_size, 0, stream>>>(output, bias, N, out_features);
    }

    cublasDestroy(handle);
}

__global__ void add_bias_kernel(float* __restrict__ output,
                                const float* __restrict__ bias,
                                int N,
                                int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_features;
    if (idx >= total) return;
    int col = idx % out_features;
    output[idx] += bias[col];
}
