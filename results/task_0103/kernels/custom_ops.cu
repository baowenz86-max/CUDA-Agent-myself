// Full content with grid-stride loop for conv1d
#include <cuda_runtime.h>
#include <cstdint>

__global__ void tile_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int64_t total_elements,
                            int64_t N,
                            int64_t C,
                            int64_t L,
                            int64_t R) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t out_batch = i / (C * L);
        int64_t in_batch = out_batch % N;
        int64_t pos_in_batch = i - out_batch * (C * L);
        int64_t src_idx = in_batch * (C * L) + pos_in_batch;
        output[i] = input[src_idx];
    }
}

__global__ void conv1d_kernel(const float* __restrict__ x,
                              const float* __restrict__ weight,
                              const float* __restrict__ bias,
                              float* __restrict__ y,
                              int N, int C_in, int C_out, int K, int L, int out_L,
                              int stride, int pad, int dilation) {
    int64_t total = (int64_t)N * C_out * out_L;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t grid_stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += grid_stride) {
        int n = i / (C_out * out_L);
        int rem = i % (C_out * out_L);
        int co = rem / out_L;
        int l_out = rem % out_L;

        float acc = 0.0f;
        if (bias != nullptr) {
            acc = bias[co];
        }
        for (int ci = 0; ci < C_in; ++ci) {
            for (int k = 0; k < K; ++k) {
                int l_in = l_out * stride + k * dilation - pad;
                if (l_in >= 0 && l_in < L) {
                    acc += x[(int64_t)n * C_in * L + ci * L + l_in] *
                           weight[(int64_t)co * C_in * K + ci * K + k];
                }
            }
        }
        y[i] = acc;
    }
}

void tile_forward_launcher(const float* input, float* output,
                           int64_t N, int64_t C, int64_t L, int64_t R,
                           cudaStream_t stream) {
    int64_t total = N * R * C * L;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    tile_kernel<<<blocks, threads, 0, stream>>>(input, output, total, N, C, L, R);
}

int conv1d_forward_launcher(const float* x, const float* weight, const float* bias,
                            float* y,
                            int N, int C_in, int C_out, int kernel_size, int L,
                            int stride, int pad, int dilation, int groups,
                            cudaStream_t stream) {
    if (groups != 1) return -1;
    int out_L = (L + 2 * pad - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t total = (int64_t)N * C_out * out_L;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    conv1d_kernel<<<blocks, threads, 0, stream>>>(
        x, weight, bias, y,
        N, C_in, C_out, kernel_size, L, out_L,
        stride, pad, dilation);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return -2;
    return 0;
}

extern "C" {
    void tile_launcher(const float* input, float* output, int64_t N, int64_t C, int64_t L, int64_t R, cudaStream_t stream) {
        tile_forward_launcher(input, output, N, C, L, R, stream);
    }
    int conv1d_launcher(const float* x, const float* weight, const float* bias, float* y,
                        int N, int C_in, int C_out, int kernel_size, int L,
                        int stride, int pad, int dilation, int groups, cudaStream_t stream) {
        return conv1d_forward_launcher(x, weight, bias, y, N, C_in, C_out, kernel_size, L,
                                       stride, pad, dilation, groups, stream);
    }
}
