// kernels/custom_ops.cu
// CUDA kernels for BatchNorm2d evaluation and clamp_min.

#include <cuda_runtime.h>

__global__ void batch_norm_eval_kernel_v2(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int64_t total_elements,
    int64_t C,
    int64_t input_stride_n, int64_t input_stride_c, int64_t input_stride_h, int64_t input_stride_w,
    int64_t output_stride_n, int64_t output_stride_c, int64_t output_stride_h, int64_t output_stride_w
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        int64_t rem = linear;
        int64_t n_coord = rem / input_stride_n;
        rem %= input_stride_n;
        int64_t c_coord = rem / input_stride_c;
        rem %= input_stride_c;
        int64_t h_coord = rem / input_stride_h;
        rem %= input_stride_h;
        int64_t w_coord = rem / input_stride_w; // input_stride_w is usually 1

        float scale, shift;
        if (weight != nullptr && bias != nullptr) {
            float inv_std = rsqrtf(running_var[c_coord] + eps);
            scale = weight[c_coord] * inv_std;
            shift = bias[c_coord] - running_mean[c_coord] * scale;
        } else if (weight != nullptr) {
            float inv_std = rsqrtf(running_var[c_coord] + eps);
            scale = weight[c_coord] * inv_std;
            shift = -running_mean[c_coord] * scale;
        } else if (bias != nullptr) {
            float inv_std = rsqrtf(running_var[c_coord] + eps);
            scale = inv_std;
            shift = bias[c_coord] - running_mean[c_coord] * scale;
        } else {
            float inv_std = rsqrtf(running_var[c_coord] + eps);
            scale = inv_std;
            shift = -running_mean[c_coord] * scale;
        }

        int64_t input_offset = n_coord * input_stride_n + c_coord * input_stride_c + h_coord * input_stride_h + w_coord * input_stride_w;
        int64_t output_offset = n_coord * output_stride_n + c_coord * output_stride_c + h_coord * output_stride_h + w_coord * output_stride_w;
        output[output_offset] = input[input_offset] * scale + shift;
    }
}

__global__ void clamp_min_kernel_v2(
    const float* __restrict__ input,
    float* __restrict__ output,
    float min_val,
    int64_t total_elements,
    int64_t input_stride_n, int64_t input_stride_c, int64_t input_stride_h, int64_t input_stride_w,
    int64_t output_stride_n, int64_t output_stride_c, int64_t output_stride_h, int64_t output_stride_w
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t linear = idx; linear < total_elements; linear += stride) {
        int64_t rem = linear;
        int64_t n_coord = rem / input_stride_n;
        rem %= input_stride_n;
        int64_t c_coord = rem / input_stride_c;
        rem %= input_stride_c;
        int64_t h_coord = rem / input_stride_h;
        rem %= input_stride_h;
        int64_t w_coord = rem / input_stride_w;

        int64_t input_offset = n_coord * input_stride_n + c_coord * input_stride_c + h_coord * input_stride_h + w_coord * input_stride_w;
        int64_t output_offset = n_coord * output_stride_n + c_coord * output_stride_c + h_coord * output_stride_h + w_coord * output_stride_w;
        output[output_offset] = fmaxf(input[input_offset], min_val);
    }
}

extern "C" void batch_norm_eval_launcher(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t input_stride_n, int64_t input_stride_c, int64_t input_stride_h, int64_t input_stride_w,
    int64_t output_stride_n, int64_t output_stride_c, int64_t output_stride_h, int64_t output_stride_w,
    cudaStream_t stream
) {
    int64_t total_elements = N * C * H * W;
    if (total_elements <= 0) return;

    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks <= 0) blocks = 1;

    batch_norm_eval_kernel_v2<<<blocks, threads, 0, stream>>>(
        input, output, weight, bias, running_mean, running_var,
        eps, total_elements, C,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        output_stride_n, output_stride_c, output_stride_h, output_stride_w
    );
}

extern "C" void clamp_min_launcher(
    const float* input,
    float* output,
    float min_val,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t input_stride_n, int64_t input_stride_c, int64_t input_stride_h, int64_t input_stride_w,
    int64_t output_stride_n, int64_t output_stride_c, int64_t output_stride_h, int64_t output_stride_w,
    cudaStream_t stream
) {
    int64_t total_elements = N * C * H * W;
    if (total_elements <= 0) return;

    int threads = 256;
    int blocks = (int)((total_elements + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks <= 0) blocks = 1;

    clamp_min_kernel_v2<<<blocks, threads, 0, stream>>>(
        input, output, min_val, total_elements,
        input_stride_n, input_stride_c, input_stride_h, input_stride_w,
        output_stride_n, output_stride_c, output_stride_h, output_stride_w
    );
}
