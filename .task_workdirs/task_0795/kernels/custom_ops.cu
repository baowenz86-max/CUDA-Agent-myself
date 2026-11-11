// CUDA kernel and launcher for custom_stack_relu
// Implements torch.stack([x] * num_repeats, dim=dim) followed by relu.
// Input: float32 CUDA tensor (logically contiguous, handled by binding)
// Output: float32 CUDA tensor with inserted dimension at dim.
// No torch/ATen dependencies.

#include <cuda_runtime.h>

// Kernel: each thread computes one output element, grid-stride loop.
// We flatten the output dimensions as: outer * num_repeats * inner,
// where outer = product of dims before dim, inner = product of dims after dim.
// For each output index, we map back to input index using the insertion rule.
__global__ void stack_relu_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int64_t total_output,
                                  int64_t outer,
                                  int64_t inner,
                                  int64_t num_repeats) {
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_output; idx += stride) {
        int64_t out_idx = idx;
        int64_t outer_idx = out_idx / (num_repeats * inner);
        int64_t rem = out_idx % (num_repeats * inner);
        // repeat_idx not needed for input mapping, but can be used if needed
        // int64_t repeat_idx = rem / inner;
        int64_t inner_idx = rem % inner;
        int64_t input_idx = outer_idx * inner + inner_idx;
        float val = input[input_idx];
        output[out_idx] = fmaxf(val, 0.0f);
    }
}

// Launcher exposed to binding layer.
// input and output must be valid CUDA pointers to float32 memory.
// outer, inner, num_repeats are derived from input shape and dim.
// total_output = outer * num_repeats * inner.
// stream is the CUDA stream to launch on.
extern "C" void custom_stack_relu_launcher(const float* input,
                                          float* output,
                                          int64_t outer,
                                          int64_t inner,
                                          int64_t num_repeats,
                                          cudaStream_t stream) {
    int64_t total_output = outer * num_repeats * inner;
    if (total_output <= 0) return;

    const int block_size = 256;
    int64_t grid_size = (total_output + block_size - 1) / block_size;
    // Limit grid size to avoid exceeding max grid dimension (2^31-1)
    if (grid_size > 2147483647LL) {
        grid_size = 2147483647LL;
    }
    stack_relu_kernel<<<(unsigned int)grid_size, block_size, 0, stream>>>(
        input, output, total_output, outer, inner, num_repeats);
}
