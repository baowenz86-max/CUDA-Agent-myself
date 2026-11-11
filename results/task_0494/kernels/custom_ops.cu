#include <cuda_runtime.h>

// Fused softplus + leaky_relu (negative_slope=0.01)
// input: float tensor, numel elements
// output: float tensor with same shape
__global__ void softplus_leaky_relu_kernel(const float* __restrict__ input,
                                           float* __restrict__ output,
                                           int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    const float neg_slope = 0.01f;
    for (int64_t i = idx; i < numel; i += stride) {
        // softplus: log(1 + exp(x)) = log1p(exp(x))
        float sp = log1pf(expf(input[i]));
        // leaky relu
        output[i] = sp >= 0.0f ? sp : neg_slope * sp;
    }
}

// C interface launcher; binding.cpp will allocate output and pass raw pointers + stream.
extern "C" void softplus_leaky_relu_launcher(const float* input,
                                              float* output,
                                              int64_t numel,
                                              cudaStream_t stream) {
    if (numel <= 0) return;
    const int block_size = 256;
    int64_t blocks = (numel + block_size - 1) / block_size;
    // grid cap to avoid too many blocks (max grid size typically 2^31-1); use grid-stride loop
    const int64_t max_blocks = 65535; // safe cap
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }
    softplus_leaky_relu_kernel<<<(int)blocks, block_size, 0, stream>>>(input, output, numel);
}
