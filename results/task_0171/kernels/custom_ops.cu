#include <cuda_runtime.h>
#include <math.h>

// Kernel 1: compute max over dim=1 for each (b,c) and store in max_vals
__global__ void log_softmax_max_kernel(
    const float* __restrict__ input,
    float* __restrict__ max_vals,
    int B, int N, int C,
    long long stride_b, long long stride_n, long long stride_c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = B * C;
    int stride = blockDim.x * gridDim.x;
    for (int pair = idx; pair < total_pairs; pair += stride) {
        int b = pair / C;
        int c = pair % C;
        const float* base = input + b * stride_b + c * stride_c;
        float max_val = -INFINITY;
        for (int n = 0; n < N; ++n) {
            float v = base[n * stride_n];
            if (v > max_val) max_val = v;
        }
        max_vals[pair] = max_val;
    }
}

// Kernel 2: compute sumexp and write log_softmax output
__global__ void log_softmax_output_kernel(
    const float* __restrict__ input,
    const float* __restrict__ max_vals,
    float* __restrict__ output,
    int B, int N, int C,
    long long in_stride_b, long long in_stride_n, long long in_stride_c,
    long long out_stride_b, long long out_stride_n, long long out_stride_c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = B * C;
    int stride = blockDim.x * gridDim.x;
    for (int pair = idx; pair < total_pairs; pair += stride) {
        int b = pair / C;
        int c = pair % C;
        const float* in_base = input + b * in_stride_b + c * in_stride_c;
        float max_val = max_vals[pair];
        float sum_exp = 0.0f;
        for (int n = 0; n < N; ++n) {
            sum_exp += __expf(in_base[n * in_stride_n] - max_val);
        }
        float log_sum = logf(sum_exp) + max_val;
        float* out_base = output + b * out_stride_b + c * out_stride_c;
        for (int n = 0; n < N; ++n) {
            out_base[n * out_stride_n] = in_base[n * in_stride_n] - log_sum;
        }
    }
}

// ELU kernel: elementwise with arbitrary strides
__global__ void elu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long total_elements,
    long long input_stride_b, long long input_stride_n, long long input_stride_c,
    long long output_stride_b, long long output_stride_n, long long output_stride_c,
    int B, int N, int C)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long linear = idx; linear < total_elements; linear += stride) {
        int b = linear / ((long long)N * C);
        int rem = linear % ((long long)N * C);
        int n = rem / C;
        int c = rem % C;
        float val = input[b * input_stride_b + n * input_stride_n + c * input_stride_c];
        float out_val = (val > 0.0f) ? val : (__expf(val) - 1.0f);
        output[b * output_stride_b + n * output_stride_n + c * output_stride_c] = out_val;
    }
}

extern "C" void log_softmax_dim1_launcher(
    const float* input,
    float* output,
    int B,
    int N,
    int C,
    long long in_stride_b,
    long long in_stride_n,
    long long in_stride_c,
    long long out_stride_b,
    long long out_stride_n,
    long long out_stride_c,
    cudaStream_t stream)
{
    int total_pairs = B * C;
    int block = 256;
    int grid = (total_pairs + block - 1) / block;
    if (grid > 65535) grid = 65535;

    float* max_vals;
    cudaMalloc(&max_vals, total_pairs * sizeof(float));
    if (max_vals == nullptr) return;

    log_softmax_max_kernel<<<grid, block, 0, stream>>>(
        input,
        max_vals,
        B, N, C,
        in_stride_b, in_stride_n, in_stride_c);

    log_softmax_output_kernel<<<grid, block, 0, stream>>>(
        input,
        max_vals,
        output,
        B, N, C,
        in_stride_b, in_stride_n, in_stride_c,
        out_stride_b, out_stride_n, out_stride_c);

    cudaFree(max_vals);
}

extern "C" void elu_launcher(
    const float* input,
    float* output,
    long long total_elements,
    int B,
    int N,
    int C,
    long long input_stride_b,
    long long input_stride_n,
    long long input_stride_c,
    long long output_stride_b,
    long long output_stride_n,
    long long output_stride_c,
    cudaStream_t stream)
{
    int block = 256;
    int grid = (int)((total_elements + block - 1) / block);
    if (grid > 65535) grid = 65535;
    elu_kernel<<<grid, block, 0, stream>>>(
        input,
        output,
        total_elements,
        input_stride_b, input_stride_n, input_stride_c,
        output_stride_b, output_stride_n, output_stride_c,
        B, N, C);
}
