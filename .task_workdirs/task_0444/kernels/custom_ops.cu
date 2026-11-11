#include <cuda_runtime.h>
#include <cmath>

// Simple elementwise sigmoid: output[i] = 1 / (1 + exp(-input[i]))
// Supports arbitrary strides.
__global__ void sigmoid_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long numel,
    long long input_stride,
    long long output_stride)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float x = input[i * input_stride];
        output[i * output_stride] = 1.0f / (1.0f + expf(-x));
    }
}

// Reduce product along dimension 1 (channels) for 4D tensor [N, C, H, W],
// output [N, 1, H, W] with keepdim=True.
// Input and output may have arbitrary strides.
__global__ void prod_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long N,
    long long C,
    long long H,
    long long W,
    long long input_stride0,
    long long input_stride1,
    long long input_stride2,
    long long input_stride3,
    long long output_stride0,
    long long output_stride1,
    long long output_stride2,
    long long output_stride3)
{
    long long total = N * H * W;
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < total; i += stride) {
        long long n = i / (H * W);
        long long rem = i % (H * W);
        long long h = rem / W;
        long long w = rem % W;

        const float* base = input + n * input_stride0 + h * input_stride2 + w * input_stride3;
        float prod = 1.0f;
        // Sequential multiplication for stability and deterministic order
        for (long long c = 0; c < C; ++c) {
            prod *= base[c * input_stride1];
        }

        float* out_ptr = output + n * output_stride0 + h * output_stride2 + w * output_stride3;
        // output_stride1 should be 0 for keepdim but we ignore it
        *out_ptr = prod;
    }
}

// Elementwise clamp: output[i] = min(max(input[i], min_val), max_val)
__global__ void clamp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long numel,
    long long input_stride,
    long long output_stride,
    float min_val,
    float max_val)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = idx; i < numel; i += stride) {
        float x = input[i * input_stride];
        x = fminf(fmaxf(x, min_val), max_val);
        output[i * output_stride] = x;
    }
}

// Helper to compute grid size, capped at 65535 in x dimension (safe maximum for all CUDA devices)
// with grid-stride loops handling the remainder.
static inline int ceildiv(long long a, long long b) {
    return (int)((a + b - 1) / b);
}

inline int clamp_grid_size(long long n, int block_size) {
    long long blocks = ceildiv(n, block_size);
    const int max_grid = 65535;
    if (blocks > max_grid) blocks = max_grid;
    return (int)blocks;
}

extern "C" void sigmoid_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    long long input_stride,
    long long output_stride,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = clamp_grid_size(numel, threads);
    sigmoid_kernel<<<blocks, threads, 0, stream>>>(input, output, numel, input_stride, output_stride);
}

extern "C" void prod_dim1_forward_launcher(
    const float* input,
    float* output,
    long long N,
    long long C,
    long long H,
    long long W,
    long long input_stride0,
    long long input_stride1,
    long long input_stride2,
    long long input_stride3,
    long long output_stride0,
    long long output_stride1,
    long long output_stride2,
    long long output_stride3,
    cudaStream_t stream)
{
    int threads = 256;
    long long total = N * H * W;
    int blocks = clamp_grid_size(total, threads);
    prod_dim1_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W,
        input_stride0, input_stride1, input_stride2, input_stride3,
        output_stride0, output_stride1, output_stride2, output_stride3);
}

extern "C" void clamp_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    long long input_stride,
    long long output_stride,
    float min_val,
    float max_val,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = clamp_grid_size(numel, threads);
    clamp_kernel<<<blocks, threads, 0, stream>>>(input, output, numel, input_stride, output_stride, min_val, max_val);
}
