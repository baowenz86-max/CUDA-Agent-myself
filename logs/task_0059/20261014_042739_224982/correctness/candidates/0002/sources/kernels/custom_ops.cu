// custom_ops.cu
// Contains CUDA kernels and launchers for custom operations.
// adaptive_avg_pool1d: custom CUDA implementation.
// conv_transpose2d: will use cuDNN; no CUDA kernel here.

#include <cuda_runtime.h>
#include <cstdint>

// -----------------------------------------------------------------------------
// adaptive_avg_pool1d
// -----------------------------------------------------------------------------
// Kernel: computes adaptive average pooling 1D over L dimension.
// Input: [N, C, L] float32
// Output: [N, C, output_size] float32
// For each output index i, compute start = floor(i * L / output_size),
// end = ceil((i + 1) * L / output_size), average elements [start, end).
// Grid-stride loop over all output elements (N*C*output_size).
__global__ void adaptive_avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N,
    int C,
    int L,
    int output_size)
{
    int total = N * C * output_size;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int pos = idx; pos < total; pos += stride) {
        int out_idx = pos % output_size;
        int tmp = pos / output_size;
        int c = tmp % C;
        int n = tmp / C;

        int start = (int)((long long)out_idx * L / output_size); // floor
        int end = (int)(((long long)(out_idx + 1) * L + output_size - 1) / output_size); // ceil
        if (end > L) end = L; // safety, though formula should yield <= L

        const float* base = input + ((long long)n * C + c) * L;
        float sum = 0.0f;
        for (int k = start; k < end; ++k) {
            sum += base[k];
        }
        float avg = (end > start) ? sum / (float)(end - start) : 0.0f;
        output[pos] = avg;
    }
}

// Launcher for adaptive_avg_pool1d
// Extern "C" so it can be called from binding.cpp
extern "C" void adaptive_avg_pool1d_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int L,
    int output_size,
    cudaStream_t stream)
{
    int total = N * C * output_size;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // cap blocks to avoid excessive grid size if total is huge; grid-stride loop handles it
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) return; // no elements
    adaptive_avg_pool1d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, L, output_size);
}

// -----------------------------------------------------------------------------
// conv_transpose2d
// -----------------------------------------------------------------------------
// This operation will be implemented using cuDNN in the binding.cpp file.
// No CUDA kernel is needed here.
// The launcher for conv_transpose2d will be provided by the C++ binding.
// We only include a note that this file may be empty for conv_transpose2d.
