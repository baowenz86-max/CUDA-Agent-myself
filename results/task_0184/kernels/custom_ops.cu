#include <cuda_runtime.h>
#include <cfloat>

// LogSoftmax over dimension 1 (channels) for 4D tensor (B, C, H, W).
// Input may be non-contiguous; provided strides define layout.
//
// Grid-stride loop over B*H*W spatial positions. Each thread handles one position.
// For each position, loop over C to compute max, then sum of exp, then write
// logsoftmax values for all C channels into output (which may be contiguous).
__global__ void log_softmax_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int H, int W,
    long long in_stride_b, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_b, long long out_stride_c, long long out_stride_h, long long out_stride_w)
{
    long long total = (long long)B * H * W;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int b = idx / ((long long)H * W);
        long long rem = idx - (long long)b * H * W;
        int h = rem / W;
        int w = rem - (long long)h * W;

        const float* in_base = input + b * in_stride_b + h * in_stride_h + w * in_stride_w;
        float* out_base = output + b * out_stride_b + h * out_stride_h + w * out_stride_w;

        // Compute max over channels
        float max_val = -FLT_MAX;
        for (int c = 0; c < C; ++c) {
            float v = in_base[c * in_stride_c];
            if (v > max_val) max_val = v;
        }

        // Compute sum of exp(x - max)
        float sum_exp = 0.0f;
        for (int c = 0; c < C; ++c) {
            float v = in_base[c * in_stride_c];
            sum_exp += __expf(v - max_val);
        }

        float log_sum = logf(sum_exp);

        for (int c = 0; c < C; ++c) {
            float v = in_base[c * in_stride_c];
            float res = (v - max_val) - log_sum;
            out_base[c * out_stride_c] = res;
        }
    }
}

// Elementwise clamp with min/max, arbitrary strides
__global__ void clamp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long numel,
    long long in_stride0, long long in_stride1, long long in_stride2, long long in_stride3,
    long long out_stride0, long long out_stride1, long long out_stride2, long long out_stride3,
    float min_val, float max_val)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < numel; idx += stride) {
        // Compute 4D coordinates from linear index assuming row-major order of logical shape B,C,H,W
        // We don't know actual dims here, but we can treat strides as long long offsets.
        // We need dims to map linear index to coordinates. Instead, we can use simple
        // linear index and compute memory offset via strides provided? But we don't have dims.
        // Simplify: clamp only supports contiguous? No, contract says arbitrary strides.
        // To handle arbitrary strides, we need dims. However, we can pass dims as well.
        // But launcher may have dims. We'll adjust kernel to take B,C,H,W.
    }
}

// Corrected clamp kernel with dims
__global__ void clamp_kernel_strided(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int H, int W,
    long long in_stride_b, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_b, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    float min_val, float max_val)
{
    long long total = (long long)B * C * H * W;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int b = idx / ((long long)C * H * W);
        long long rem = idx - (long long)b * C * H * W;
        int c = rem / ((long long)H * W);
        rem = rem - (long long)c * H * W;
        int h = rem / W;
        int w = rem - (long long)h * W;

        const float* in_ptr = input + b * in_stride_b + c * in_stride_c + h * in_stride_h + w * in_stride_w;
        float* out_ptr = output + b * out_stride_b + c * out_stride_c + h * out_stride_h + w * out_stride_w;
        float v = *in_ptr;
        *out_ptr = fminf(fmaxf(v, min_val), max_val);
    }
}

// Launcher for log_softmax_forward: input and output are raw pointers, strides in elements.
extern "C" void log_softmax_forward_launcher(
    const float* input,
    float* output,
    int B, int C, int H, int W,
    long long in_stride_b, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_b, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream)
{
    long long total = (long long)B * H * W;
    int block_size = 256;
    long long grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;

    log_softmax_dim1_kernel<<<(int)grid_size, block_size, 0, stream>>>(
        input, output, B, C, H, W,
        in_stride_b, in_stride_c, in_stride_h, in_stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w);
}

// Launcher for clamp_forward
extern "C" void clamp_forward_launcher(
    const float* input,
    float* output,
    int B, int C, int H, int W,
    long long in_stride_b, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_b, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    float min_val, float max_val,
    cudaStream_t stream)
{
    long long total = (long long)B * C * H * W;
    int block_size = 256;
    long long grid_size = (total + block_size - 1) / block_size;
    if (grid_size > 65535) grid_size = 65535;

    clamp_kernel_strided<<<(int)grid_size, block_size, 0, stream>>>(
        input, output, B, C, H, W,
        in_stride_b, in_stride_c, in_stride_h, in_stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w,
        min_val, max_val);
}
