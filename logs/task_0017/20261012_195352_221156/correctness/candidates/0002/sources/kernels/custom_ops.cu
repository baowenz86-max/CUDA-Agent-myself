#include <cuda_runtime.h>
#include <cmath>

// ConvTranspose1d kernel (float32 only, as used by this task)
// Input: (batch, in_channels, in_len)
// Weight: (in_channels, out_channels, kernel_size)
// Bias: (out_channels)
// Output: (batch, out_channels, out_len)
// out_len = (in_len - 1) * stride - 2 * padding + kernel_size + output_padding
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch,
    int in_channels,
    int out_channels,
    int in_len,
    int out_len,
    int kernel_size,
    int stride,
    int padding,
    int output_padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_len;
    if (idx >= total) return;

    int ow = idx % out_len;
    int oc = (idx / out_len) % out_channels;
    int n = idx / (out_len * out_channels);

    float acc = bias[oc];

    // For each input position that can contribute to this output position:
    // ow = iw * stride - padding + k, for k in [0, kernel_size-1]
    // So iw = (ow + padding - k) / stride, must be integer and in [0, in_len-1]
    for (int k = 0; k < kernel_size; ++k) {
        int numerator = ow + padding - k;
        if (numerator % stride != 0) continue;
        int iw = numerator / stride;
        if (iw < 0 || iw >= in_len) continue;

        for (int ic = 0; ic < in_channels; ++ic) {
            // Weight index: (ic, oc, k)
            float w = weight[(ic * out_channels + oc) * kernel_size + k];
            float x = input[(n * in_channels + ic) * in_len + iw];
            acc += x * w;
        }
    }

    output[idx] = acc;
}

// Min scalar kernel: output = min(input, scalar)
__global__ void min_scalar_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    float scalar,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float v = input[idx];
    output[idx] = v < scalar ? v : scalar;
}

// Erf kernel: output = erf(input)
__global__ void erf_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    output[idx] = erff(input[idx]);
}

// Launchers (extern "C" for binding)

extern "C" void conv_transpose1d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int in_len,
    int out_len,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream
) {
    int total = batch * out_channels * out_len;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch, in_channels, out_channels, in_len, out_len,
        kernel_size, stride, padding, output_padding
    );
}

extern "C" void min_scalar_forward_launcher(
    const float* input,
    float* output,
    int total,
    float scalar,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    min_scalar_kernel<<<blocks, threads, 0, stream>>>(input, output, scalar, total);
}

extern "C" void erf_forward_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    erf_kernel<<<blocks, threads, 0, stream>>>(input, output, total);
}
