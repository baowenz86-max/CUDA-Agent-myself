#include <cuda_runtime.h>

__global__ void relu_forward_kernel(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; idx < numel; idx += stride) {
        float val = input[idx];
        output[idx] = (val > 0.0f) ? val : 0.0f;
    }
}

template <int UPSACALE_FACTOR>
__global__ void pixel_shuffle_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C_out, int64_t H_in, int64_t W_in,
    int64_t H_out, int64_t W_out) {
    constexpr int r = UPSACALE_FACTOR;
    const int64_t C_in = C_out * r * r;
    const int64_t out_plane = H_out * W_out;
    const int64_t out_ch_stride = C_out * out_plane;
    const int64_t in_plane = H_in * W_in;
    const int64_t in_ch_stride = C_in * in_plane;
    const int64_t total = N * out_ch_stride;

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; idx < total; idx += stride) {
        int64_t n = idx / out_ch_stride;
        int64_t rem = idx - n * out_ch_stride;
        int64_t c = rem / out_plane;
        int64_t rem2 = rem - c * out_plane;
        int64_t h_out = rem2 / W_out;
        int64_t w_out = rem2 - h_out * W_out;

        int64_t h_in = h_out / r;
        int64_t i = h_out - h_in * r;
        int64_t w_in = w_out / r;
        int64_t j = w_out - w_in * r;

        int64_t input_c = c * r * r + i * r + j;
        int64_t input_idx = n * in_ch_stride + input_c * in_plane + h_in * W_in + w_in;

        output[idx] = input[input_idx];
    }
}

__global__ void deconv_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t N, int64_t C_in, int64_t H_in, int64_t W_in,
    int64_t C_out, int64_t kH, int64_t kW,
    int64_t H_out, int64_t W_out,
    int64_t stride, int64_t padding, int64_t output_padding,
    int64_t total_output) {

    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride_thread = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; idx < total_output; idx += stride_thread) {
        int64_t n = idx / (C_out * H_out * W_out);
        int64_t rem = idx - n * C_out * H_out * W_out;
        int64_t c_out = rem / (H_out * W_out);
        int64_t rem2 = rem - c_out * H_out * W_out;
        int64_t h_out = rem2 / W_out;
        int64_t w_out = rem2 - h_out * W_out;

        float val = (bias != nullptr) ? bias[c_out] : 0.0f;

        for (int64_t c_in = 0; c_in < C_in; ++c_in) {
            for (int64_t kh = 0; kh < kH; ++kh) {
                for (int64_t kw = 0; kw < kW; ++kw) {
                    int64_t h_in = h_out + padding - kh;
                    int64_t w_in = w_out + padding - kw;
                    if (h_in < 0 || w_in < 0 || h_in >= H_in * stride || w_in >= W_in * stride) continue;
                    if (h_in % stride != 0 || w_in % stride != 0) continue;
                    h_in /= stride;
                    w_in /= stride;
                    if (h_in >= H_in || w_in >= W_in) continue;

                    int64_t input_idx = ((n * C_in + c_in) * H_in + h_in) * W_in + w_in;
                    int64_t weight_idx = ((c_in * C_out + c_out) * kH + kh) * kW + kw;
                    val += input[input_idx] * weight[weight_idx];
                }
            }
        }
        output[idx] = val;
    }
}

extern "C" void relu_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream) {
    if (numel <= 0) return;
    const int block = 256;
    int64_t grid64 = (numel + block - 1) / block;
    if (grid64 > 65535) grid64 = 65535;
    int grid = static_cast<int>(grid64);
    relu_forward_kernel<<<grid, block, 0, stream>>>(input, output, numel);
}

extern "C" void pixel_shuffle_forward_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C_out,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int upscale_factor,
    cudaStream_t stream) {
    if (N <= 0 || C_out <= 0 || H_in <= 0 || W_in <= 0 || H_out <= 0 || W_out <= 0) return;
    const int block = 256;
    int64_t total = N * C_out * H_out * W_out;
    int64_t grid64 = (total + block - 1) / block;
    if (grid64 > 65535) grid64 = 65535;
    int grid = static_cast<int>(grid64);

    if (upscale_factor == 2) {
        pixel_shuffle_forward_kernel<2><<<grid, block, 0, stream>>>(
            input, output, N, C_out, H_in, W_in, H_out, W_out);
    } else if (upscale_factor == 3) {
        pixel_shuffle_forward_kernel<3><<<grid, block, 0, stream>>>(
            input, output, N, C_out, H_in, W_in, H_out, W_out);
    } else if (upscale_factor == 4) {
        pixel_shuffle_forward_kernel<4><<<grid, block, 0, stream>>>(
            input, output, N, C_out, H_in, W_in, H_out, W_out);
    } else {
        // Unsupported factor: do nothing (binding layer should have checked)
    }
}

extern "C" void deconv_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t N, int64_t C_in, int64_t H_in, int64_t W_in,
    int64_t C_out, int64_t kH, int64_t kW,
    int64_t H_out, int64_t W_out,
    int64_t stride, int64_t padding, int64_t output_padding,
    cudaStream_t stream) {
    if (N <= 0 || C_in <= 0 || H_in <= 0 || W_in <= 0 || C_out <= 0 || H_out <= 0 || W_out <= 0) return;
    int64_t total_output = N * C_out * H_out * W_out;
    const int block = 256;
    int64_t grid64 = (total_output + block - 1) / block;
    if (grid64 > 65535) grid64 = 65535;
    int grid = static_cast<int>(grid64);
    deconv_forward_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, H_in, W_in,
        C_out, kH, kW,
        H_out, W_out,
        stride, padding, output_padding,
        total_output);
}
