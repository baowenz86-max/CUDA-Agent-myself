// CUDA kernel for fused max reduction along dim=1 and division by scalar.
// Input: float32 tensor of shape [N, C, H, W], contiguous.
// Output: values (float32) shape [N, 1, H, W] and indices (int64) shape [N, 1, H, W].
// The launcher is a plain C function with raw pointers and scalar arguments,
// suitable for binding from a separate .cpp file.

#include <cuda_runtime.h>
#include <stdint.h>

// Kernel: each thread handles one (n, h, w) position.
// It reduces over C channels to find max value and its index.
__global__ void max_reduce_div_kernel(
    const float* __restrict__ input,
    float* __restrict__ values_out,
    int64_t* __restrict__ indices_out,
    int N,
    int C,
    int H,
    int W,
    float divisor)
{
    int total_spatial = N * H * W;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int pos = tid; pos < total_spatial; pos += stride) {
        int n = pos / (H * W);
        int residual = pos % (H * W);
        int h = residual / W;
        int w = residual % W;

        const float* in_ptr = input + n * C * H * W + h * W + w;
        float max_val = in_ptr[0];
        int64_t max_idx = 0;
        for (int c = 1; c < C; ++c) {
            float val = in_ptr[c * H * W];
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
        }

        values_out[n * H * W + h * W + w] = max_val / divisor;
        indices_out[n * H * W + h * W + w] = max_idx;
    }
}

// Launcher function (extern "C" to avoid name mangling)
extern "C" void max_reduce_div_launcher(
    const float* input,
    float* values_out,
    int64_t* indices_out,
    int N,
    int C,
    int H,
    int W,
    double divisor,
    cudaStream_t stream)
{
    float divisor_f = static_cast<float>(divisor);
    int total_spatial = N * H * W;
    int threads = 256;
    int blocks = (total_spatial + threads - 1) / threads;

    max_reduce_div_kernel<<<blocks, threads, 0, stream>>>(
        input, values_out, indices_out, N, C, H, W, divisor_f);
}
