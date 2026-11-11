#include <cuda_runtime.h>
#include <cmath>

// SELU constants (inlined to avoid undefined symbol issues)
constexpr float SELU_SCALE = 1.0507009873554804934193349852946f;
constexpr float SELU_ALPHA = 1.6732632423543772848170429916717f;

// Elementwise SELU kernel
// Each thread computes one output element.
// Input and output are strided 2D tensors with shape [rows, cols].
// We flatten the index and map to (row, col) using output strides.
__global__ void selu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols,
    long long input_stride0,
    long long input_stride1,
    long long output_stride0,
    long long output_stride1)
{
    long long total = (long long)rows * cols;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long i = idx; i < total; i += stride) {
        int row = i / cols;
        int col = i % cols;
        float x = input[row * input_stride0 + col * input_stride1];
        float out;
        if (x >= 0.0f) {
            out = SELU_SCALE * x;
        } else {
            out = SELU_SCALE * (SELU_ALPHA * expf(x) - SELU_ALPHA);
        }
        output[row * output_stride0 + col * output_stride1] = out;
    }
}

// Cumulative sum along dimension 0.
// Simplest correct approach: use one thread per column, each thread loops over rows.
// This is correct and sufficient for small cols. If cols > number of blocks, we use grid-stride.
__global__ void cumsum0_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols,
    long long input_stride0,
    long long input_stride1,
    long long output_stride0,
    long long output_stride1)
{
    // One thread per column, using block and thread mapping: blockIdx.x corresponds to column, threadIdx.x must be 0.
    int col = blockIdx.x;
    if (col >= cols) return;

    if (threadIdx.x != 0) return; // Only thread 0 in each block does work

    float running_sum = 0.0f;
    for (int r = 0; r < rows; ++r) {
        running_sum += input[r * input_stride0 + col * input_stride1];
        output[r * output_stride0 + col * output_stride1] = running_sum;
    }
}

// Launcher for SELU
// In launcher, use 256 threads and grid-stride loop as before.
extern "C" void selu_launcher(
    const float* input,
    float* output,
    int rows,
    int cols,
    long long input_stride0,
    long long input_stride1,
    long long output_stride0,
    long long output_stride1,
    cudaStream_t stream)
{
    long long total = (long long)rows * cols;
    int threads = 256;
    long long blocks_ll = (total + threads - 1) / threads;
    int max_blocks = 65535;
    int blocks = (int)min(blocks_ll, (long long)max_blocks);
    selu_kernel<<<blocks, threads, 0, stream>>>(
        input, output, rows, cols,
        input_stride0, input_stride1,
        output_stride0, output_stride1);
}

// Launcher for cumsum0
// Launch exactly cols blocks, each with 1 thread.
extern "C" void cumsum0_launcher(
    const float* input,
    float* output,
    int rows,
    int cols,
    long long input_stride0,
    long long input_stride1,
    long long output_stride0,
    long long output_stride1,
    cudaStream_t stream)
{
    int threads = 1;
    int blocks = cols;
    cumsum0_kernel<<<blocks, threads, 0, stream>>>(
        input, output, rows, cols,
        input_stride0, input_stride1,
        output_stride0, output_stride1);
}
