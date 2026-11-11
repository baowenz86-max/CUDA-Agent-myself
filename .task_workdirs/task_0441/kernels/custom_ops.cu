#include <cuda_runtime.h>

// Kernel for BatchNorm2d evaluation mode: y = weight * (x - mean) / sqrt(var + eps) + bias
// Handles arbitrary strides; channel dimension is dim1, spatial dims are dim2 and dim3.
// weight and bias may be null if not provided; if weight is null, no scale is applied; if bias is null, no shift.
__global__ void batchnorm2d_eval_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N, int C, int H, int W,
    long long input_stride_n, long long input_stride_c, long long input_stride_h, long long input_stride_w,
    long long output_stride_n, long long output_stride_c, long long output_stride_h, long long output_stride_w,
    int has_weight, int has_bias
) {
    long long total = (long long)N * C * H * W;
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < total; i += stride) {
        // Compute n,c,h,w from flattened index using output strides (assume output same shape as input)
        int n = i / ((long long)C * H * W);
        long long rem = i % ((long long)C * H * W);
        int c = rem / ((long long)H * W);
        rem = rem % ((long long)H * W);
        int h = rem / W;
        int w = rem % W;

        // Input index using input strides
        long long in_idx = (long long)n * input_stride_n + (long long)c * input_stride_c + (long long)h * input_stride_h + (long long)w * input_stride_w;
        float x = input[in_idx];

        float mean = running_mean[c];
        float var = running_var[c];
        float inv_std = rsqrtf(var + eps);
        float normalized = (x - mean) * inv_std;

        float result = normalized;
        if (has_weight) {
            result *= weight[c];
        }
        if (has_bias) {
            result += bias[c];
        }

        // Output index using output strides
        long long out_idx = (long long)n * output_stride_n + (long long)c * output_stride_c + (long long)h * output_stride_h + (long long)w * output_stride_w;
        output[out_idx] = result;
    }
}

// Kernel for ReLU: y = max(x, 0), arbitrary strides
__global__ void relu_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long total,
    long long input_stride_n, long long input_stride_c, long long input_stride_h, long long input_stride_w,
    long long output_stride_n, long long output_stride_c, long long output_stride_h, long long output_stride_w,
    int N, int C, int H, int W
) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < total; i += stride) {
        int n = i / ((long long)C * H * W);
        long long rem = i % ((long long)C * H * W);
        int c = rem / ((long long)H * W);
        rem = rem % ((long long)H * W);
        int h = rem / W;
        int w = rem % W;

        long long in_idx = (long long)n * input_stride_n + (long long)c * input_stride_c + (long long)h * input_stride_h + (long long)w * input_stride_w;
        float x = input[in_idx];
        float result = fmaxf(x, 0.0f);

        long long out_idx = (long long)n * output_stride_n + (long long)c * output_stride_c + (long long)h * output_stride_h + (long long)w * output_stride_w;
        output[out_idx] = result;
    }
}

// Launcher for BatchNorm2d evaluation
// Inputs: output tensor (allocated by binding), input, weight (may be null), bias (may be null), running_mean, running_var, eps
// All pointers are raw device pointers, strides are in elements, N,C,H,W give shape.
extern "C" void batchnorm2d_eval_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int N, int C, int H, int W,
    long long input_stride_n, long long input_stride_c, long long input_stride_h, long long input_stride_w,
    long long output_stride_n, long long output_stride_c, long long output_stride_h, long long output_stride_w,
    int has_weight, int has_bias,
    cudaStream_t stream
) {
    dim3 block(256);
    long long total = (long long)N * C * H * W;
    int grid = (int)((total + block.x - 1) / block.x);
    // Ensure grid doesn't exceed max grid size; if too large, use grid-stride loop with max grid.
    if (grid > 65535) grid = 65535;
    batchnorm2d_eval_kernel<<<grid, block, 0, stream>>>(
        output, input, weight, bias, running_mean, running_var, (float)eps,
        N, C, H, W,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        output_stride_n, output_stride_c, output_stride_h, output_stride_w,
        has_weight, has_bias
    );
}

// Launcher for ReLU
extern "C" void relu_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    long long input_stride_n, long long input_stride_c, long long input_stride_h, long long input_stride_w,
    long long output_stride_n, long long output_stride_c, long long output_stride_h, long long output_stride_w,
    cudaStream_t stream
) {
    dim3 block(256);
    long long total = (long long)N * C * H * W;
    int grid = (int)((total + block.x - 1) / block.x);
    if (grid > 65535) grid = 65535;
    relu_kernel<<<grid, block, 0, stream>>>(
        output, input, total,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        output_stride_n, output_stride_c, output_stride_h, output_stride_w,
        N, C, H, W
    );
}
