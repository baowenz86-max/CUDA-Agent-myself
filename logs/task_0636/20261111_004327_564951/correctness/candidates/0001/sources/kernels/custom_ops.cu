#include <cuda_runtime.h>
#include <math_functions.h>

// MaxPool1d kernel: 计算窗口内最大值
__global__ void maxpool1d_kernel(const float* input, float* output,
                                 int N, int C, int L, int L_out,
                                 int kernel_size, int stride, int padding) {
    int total_output = N * C * L_out;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_total = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_output; idx += stride_total) {
        int l_out = idx % L_out;
        int nc = idx / L_out;
        int c = nc % C;
        int n = nc / C;

        int l_start = l_out * stride - padding;
        float max_val = -INFINITY;
        for (int k = 0; k < kernel_size; ++k) {
            int l_in = l_start + k;
            if (l_in >= 0 && l_in < L) {
                float val = input[((size_t)n * C + c) * L + l_in];
                if (val > max_val) max_val = val;
            }
        }
        output[idx] = max_val;
    }
}

// Tanh kernel: 逐元素计算 tanh
__global__ void tanh_kernel(const float* input, float* output, size_t total) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = tid; i < total; i += stride) {
        output[i] = tanhf(input[i]);
    }
}

// Launcher for MaxPool1d forward
extern "C" void maxpool1d_forward_launcher(
    const float* input, float* output,
    int N, int C, int L, int L_out,
    int kernel_size, int stride, int padding,
    cudaStream_t stream) {

    int total_output = N * C * L_out;
    int block_size = 256;
    int grid_size = (total_output + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535; // 限制 grid size，实际使用 grid-stride loop 处理

    maxpool1d_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, N, C, L, L_out, kernel_size, stride, padding);
}

// Launcher for Tanh forward
extern "C" void tanh_forward_launcher(
    const float* input, float* output,
    size_t total,
    cudaStream_t stream) {

    int block_size = 256;
    size_t grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;

    tanh_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, total);
}
