#include <cuda_runtime.h>

// Kernel for element-wise multiplication with per-channel bias broadcasting.
// input: [batch_size, out_channels, length_out]
// bias:   [out_channels, 1]
// output: same shape as input
// The bias is broadcast along channel dimension (axis=1), so for each element
// at (n, c, l) we multiply input[n,c,l] by bias[c].
__global__ void mul_bias_kernel(const float* __restrict__ input,
                                const float* __restrict__ bias,
                                float* __restrict__ output,
                                int batch_size,
                                int out_channels,
                                int length_out) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = (int64_t)batch_size * out_channels * length_out;
    if (idx >= total_elements) return;

    // Compute (n, c, l) from flat index
    int l = idx % length_out;
    int64_t tmp = idx / length_out;
    int c = tmp % out_channels;
    // int n = tmp / out_channels; // not needed since bias only depends on c

    output[idx] = input[idx] * bias[c];
}

// Launcher for mul_bias_forward
// input: device pointer to input tensor (float, CUDA)
// bias:  device pointer to bias tensor (float, CUDA)
// output: device pointer to output tensor (preallocated, same shape as input)
// batch_size, out_channels, length_out: dimensions of the input tensor
// stream: CUDA stream to launch the kernel on
extern "C" void mul_bias_forward_launcher(const float* input,
                                         const float* bias,
                                         float* output,
                                         int batch_size,
                                         int out_channels,
                                         int length_out,
                                         cudaStream_t stream) {
    int64_t total_elements = (int64_t)batch_size * out_channels * length_out;
    const int block_size = 256;
    int64_t grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 0) {
        mul_bias_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
            input, bias, output, batch_size, out_channels, length_out);
    }
}
