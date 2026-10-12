#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

// Kernel to repeat interleave along dimension 1 (sequence dimension).
// Input tensor is assumed to be of shape [B, S, E] contiguous,
// with elements of type T (float, half, bfloat16, double).
// Output is [B, S * repeats, E].
// Each thread handles one output element.
template <typename T>
__global__ void repeat_interleave_kernel(
    T* __restrict__ output,
    const T* __restrict__ input,
    int B, int S, int E, int repeats)
{
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = B * S * repeats * E;
    if (out_idx >= total_out) return;

    // Decompose output index
    int e = out_idx % E;
    int tmp = out_idx / E;            // [0, B * S * repeats)
    int s_rep = tmp % (S * repeats);
    int b = tmp / (S * repeats);
    int s = s_rep / repeats;

    int in_idx = ((b * S) + s) * E + e;
    output[out_idx] = input[in_idx];
}

// Host launcher for repeat_interleave along dim=1.
// Accepts raw pointers, dimensions, repeats, dtype size, and stream.
extern "C" void repeat_interleave_launcher(
    void* output,
    const void* input,
    int B, int S, int E, int repeats,
    int dtype_size,
    cudaStream_t stream)
{
    int total_out = B * S * repeats * E;
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;

    // Dispatch based on dtype_size (bytes per element)
    switch (dtype_size) {
        case 2: // half or bfloat16
            repeat_interleave_kernel<__half><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<__half*>(output),
                reinterpret_cast<const __half*>(input),
                B, S, E, repeats);
            break;
        case 4: // float
            repeat_interleave_kernel<float><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<float*>(output),
                reinterpret_cast<const float*>(input),
                B, S, E, repeats);
            break;
        case 8: // double
            repeat_interleave_kernel<double><<<blocks, threads, 0, stream>>>(
                reinterpret_cast<double*>(output),
                reinterpret_cast<const double*>(input),
                B, S, E, repeats);
            break;
        default:
            // Should not happen if binding checks dtype.
            break;
    }
}