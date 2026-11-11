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
    int C,
    int H,
    int W,
    float divisor)
{
    // Total number of spatial positions: N * H * W
    int N = gridDim.x; // one block per batch? No, gridDim.x is number of blocks, not N.
    // Instead, compute global thread index.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_spatial = N * H * W; // need N, but we didn't pass N directly. We can compute N from grid? not reliable.
    // We'll pass N explicitly in the launcher and store it in a special way? Better to pass N as a separate argument.
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
    double divisor,   // divisor as double, but kernel expects float; convert before launch
    cudaStream_t stream)
{
    // Cast divisor to float for kernel
    float divisor_f = static_cast<float>(divisor);

    int total_spatial = N * H * W;
    int threads = 256;
    int blocks = (total_spatial + threads - 1) / threads;

    max_reduce_div_kernel<<<blocks, threads, 0, stream>>>(
        input, values_out, indices_out, N, C, H, W, divisor_f);
}
