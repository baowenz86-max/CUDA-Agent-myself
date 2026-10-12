#include <cuda_runtime.h>
#include <math.h>

// Helper for erf approximation using CUDA erff (should match torch.erf within tolerance)
__device__ float erf_approx(float x) {
    return erff(x);
}

// ConvTranspose1d kernel: standard transposed convolution.
// input: [N, C_in, L_in], weight: [C_in, C_out, K], bias: [C_out]
// output: [N, C_out, L_out]
// L_out = (L_in - 1) * stride - 2 * padding + K + output_padding
// Each output element is sum over ic, kw of input[n, ic, iw] * weight[ic, oc, kw] where
// iw = (ow + padding - kw) / stride if divisible and 0 <= iw < L_in, with ow in [0, L_out-1].
__global__ void conv_transpose1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int K, int stride, int padding, int output_padding)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * L_out;
    if (idx >= total) return;

    int ow = idx % L_out;
    int oc = (idx / L_out) % C_out;
    int n = idx / (L_out * C_out);

    float sum = bias[oc];
    for (int ic = 0; ic < C_in; ++ic) {
        for (int kw = 0; kw < K; ++kw) {
            int t = ow + padding - kw;
            if (t % stride != 0) continue;
            int iw = t / stride;
            if (iw >= 0 && iw < L_in) {
                sum += input[((n * C_in + ic) * L_in) + iw] * weight[((ic * C_out + oc) * K) + kw];
            }
        }
    }
    output[idx] = sum;
}

// Min with scalar: output = min(input, val)
__global__ void min_scalar_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size,
    float val)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = fminf(x, val);
    }
}

// Erf: output = erf(input)
__global__ void erf_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = erff(input[idx]);
    }
}

// Launchers
extern "C" void conv_transpose1d_custom_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int C_out, int L_in, int L_out,
    int K, int stride, int padding, int output_padding,
    cudaStream_t stream)
{
    int total = N * C_out * L_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv_transpose1d_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        N, C_in, C_out, L_in, L_out,
        K, stride, padding, output_padding);
}

extern "C" void min_scalar_custom_launcher(
    const float* input,
    float* output,
    int size,
    float val,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    min_scalar_kernel<<<blocks, threads, 0, stream>>>(input, output, size, val);
}

extern "C" void erf_custom_launcher(
    const float* input,
    float* output,
    int size,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    erf_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
}
