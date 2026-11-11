// kernels/custom_ops.cu
#include <cuda_runtime.h>

__global__ void tanh_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = tanhf(input[i]);
    }
}

extern "C" void tanh_forward_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      cudaStream_t stream) {
    if (numel <= 0) return;
    const int block_size = 256;
    int64_t grid_size = (numel + block_size - 1) / block_size;
    const int64_t max_grid_x = 65535;
    if (grid_size > max_grid_x) grid_size = max_grid_x;
    tanh_kernel<<<(int)grid_size, block_size, 0, stream>>>(input, output, numel);
}

// Simple 3D convolution kernel (no padding, stride=1, dilation=1 assumed)
// Will be generalized in binding by adjusting index mapping or using separate kernels.
// For simplicity, assume padding=0, stride=1, dilation=1 as in task contract.
__global__ void conv3d_forward_kernel(const float* __restrict__ input,
                                      const float* __restrict__ weight,
                                      const float* __restrict__ bias,
                                      float* __restrict__ output,
                                      int N, int C_in, int D_in, int H_in, int W_in,
                                      int C_out, int KD, int KH, int KW,
                                      int D_out, int H_out, int W_out,
                                      bool has_bias) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * D_out * H_out * W_out;
    if (idx >= total) return;

    int w = idx % W_out;
    int h = (idx / W_out) % H_out;
    int d = (idx / (W_out * H_out)) % D_out;
    int c_out = (idx / (W_out * H_out * D_out)) % C_out;
    int n = idx / (W_out * H_out * D_out * C_out);

    float sum = 0.0f;
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    int in_d = d + kd;
                    int in_h = h + kh;
                    int in_w = w + kw;
                    if (in_d < D_in && in_h < H_in && in_w < W_in) {
                        int input_idx = (((n * C_in + c_in) * D_in + in_d) * H_in + in_h) * W_in + in_w;
                        int weight_idx = ((((c_out * C_in + c_in) * KD + kd) * KH + kh) * KW + kw);
                        sum += input[input_idx] * weight[weight_idx];
                    }
                }
            }
        }
    }
    if (has_bias) sum += bias[c_out];
    output[idx] = sum;
}

extern "C" void conv3d_forward_launcher(const float* input,
                                        const float* weight,
                                        const float* bias,
                                        float* output,
                                        int N, int C_in, int D_in, int H_in, int W_in,
                                        int C_out, int KD, int KH, int KW,
                                        int D_out, int H_out, int W_out,
                                        bool has_bias,
                                        cudaStream_t stream) {
    int total = N * C_out * D_out * H_out * W_out;
    if (total <= 0) return;
    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    conv3d_forward_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, D_in, H_in, W_in,
        C_out, KD, KH, KW,
        D_out, H_out, W_out,
        has_bias);
}