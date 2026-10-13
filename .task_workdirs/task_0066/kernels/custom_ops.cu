#include <cuda_runtime.h>
#include <cstdint>

// Element-wise greater-than-or-equal kernel: output[i] = (input[i] >= threshold)
__global__ void ge_kernel(const float* __restrict__ input,
                          bool* __restrict__ output,
                          int64_t numel,
                          float threshold) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = (input[i] >= threshold);
    }
}

// Element-wise ReLU6 kernel: output[i] = min(max(input[i], 0), 6)
__global__ void relu6_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        float val = input[i];
        val = fminf(fmaxf(val, 0.0f), 6.0f);
        output[i] = val;
    }
}

// Element-wise multiplication kernel: output[i] = input[i] * mask[i]
__global__ void mul_kernel(const float* __restrict__ input,
                           const float* __restrict__ mask,
                           float* __restrict__ output,
                           int64_t numel) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = input[i] * mask[i];
    }
}

// ConvTranspose1d kernel: output[b, oc, ol] = bias[oc] + sum_{ic, k} input[b, ic, il] * weight[ic, oc, k]
// where il = (ol + padding - k) / stride, if (ol + padding - k) % stride == 0 and 0 <= il < input_length.
__global__ void conv_transpose1d_kernel(const float* __restrict__ input,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias,
                                        float* __restrict__ output,
                                        int batch,
                                        int in_channels,
                                        int out_channels,
                                        int input_length,
                                        int kernel_size,
                                        int stride,
                                        int padding,
                                        int output_length) {
    int64_t total_elements = static_cast<int64_t>(batch) * out_channels * output_length;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride_loop = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (int64_t linear_idx = idx; linear_idx < total_elements; linear_idx += stride_loop) {
        int b = linear_idx / (out_channels * output_length);
        int rem = linear_idx % (out_channels * output_length);
        int oc = rem / output_length;
        int ol = rem % output_length;
        float acc = bias[oc];
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int k = 0; k < kernel_size; ++k) {
                int il = (ol + padding - k) / stride;
                if ((ol + padding - k) % stride == 0 && il >= 0 && il < input_length) {
                    acc += input[(static_cast<int64_t>(b) * in_channels + ic) * input_length + il] *
                           weight[(static_cast<int64_t>(ic) * out_channels + oc) * kernel_size + k];
                }
            }
        }
        output[linear_idx] = acc;
    }
}

// Helper to get a reasonably sized grid
static inline int get_grid_size(int64_t numel, int block_size) {
    int64_t blocks = (numel + block_size - 1) / block_size;
    if (blocks > 65535) {
        blocks = 65535;
    }
    return static_cast<int>(blocks);
}

// Launcher for ge_forward
__global__ void ge_forward_launcher_kernel(const float* input, bool* output, int64_t numel, float threshold) {}

extern "C" void ge_forward_launcher(const float* input,
                                    bool* output,
                                    int64_t numel,
                                    float threshold,
                                    cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = get_grid_size(numel, block_size);
    ge_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel, threshold);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
#ifdef DEBUG
        printf("CUDA error in ge_forward_launcher: %s\n", cudaGetErrorString(err));
#endif
    }
}

// Launcher for relu6_forward
extern "C" void relu6_forward_launcher(const float* input,
                                       float* output,
                                       int64_t numel,
                                       cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = get_grid_size(numel, block_size);
    relu6_kernel<<<grid_size, block_size, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
#ifdef DEBUG
        printf("CUDA error in relu6_forward_launcher: %s\n", cudaGetErrorString(err));
#endif
    }
}

// Launcher for mul_forward
extern "C" void mul_forward_launcher(const float* input,
                                     const float* mask,
                                     float* output,
                                     int64_t numel,
                                     cudaStream_t stream) {
    constexpr int block_size = 256;
    int grid_size = get_grid_size(numel, block_size);
    mul_kernel<<<grid_size, block_size, 0, stream>>>(input, mask, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
#ifdef DEBUG
        printf("CUDA error in mul_forward_launcher: %s\n", cudaGetErrorString(err));
#endif
    }
}

// Launcher for conv_transpose1d_forward
extern "C" void conv_transpose1d_forward_launcher(const float* input,
                                                 const float* weight,
                                                 const float* bias,
                                                 float* output,
                                                 int batch,
                                                 int in_channels,
                                                 int out_channels,
                                                 int input_length,
                                                 int kernel_size,
                                                 int stride,
                                                 int padding,
                                                 int output_length,
                                                 cudaStream_t stream) {
    constexpr int block_size = 256;
    int64_t total_elements = static_cast<int64_t>(batch) * out_channels * output_length;
    int grid_size = get_grid_size(total_elements, block_size);
    conv_transpose1d_kernel<<<grid_size, block_size, 0, stream>>>(
        input, weight, bias, output,
        batch, in_channels, out_channels, input_length, kernel_size,
        stride, padding, output_length);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
#ifdef DEBUG
        printf("CUDA error in conv_transpose1d_forward_launcher: %s\n", cudaGetErrorString(err));
#endif
    }
}
