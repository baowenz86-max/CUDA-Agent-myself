// Custom CUDA kernels for abs(ReLU6) fusion
#include <cuda_runtime.h>

// Elementwise abs then ReLU6: min(max(abs(x), 0), 6)
// input/output are float32 CUDA pointers, contiguous or arbitrary stride handled by caller
// numel: total number of elements
__global__ void abs_relu6_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 long long numel) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < numel; idx += stride) {
        float val = fabsf(input[idx]);
        val = fminf(fmaxf(val, 0.0f), 6.0f);
        output[idx] = val;
    }
}

// Launcher: no PyTorch dependencies, takes raw pointers and CUDA stream
// Used by the C++ binding layer
extern "C" void abs_relu6_forward_launcher(const float* input,
                                          float* output,
                                          long long numel,
                                          cudaStream_t stream) {
    const int block_size = 256;
    long long blocks_ll = (numel + block_size - 1) / block_size;
    // Cap blocks to avoid exceeding grid dimension limit (2^31-1); kernel uses grid-stride loop,
    // so sufficient blocks are checked. For typical numel <= 2^31, blocks_ll is within limit.
    int blocks = (blocks_ll > 2147483647LL) ? 2147483647 : (int)blocks_ll;
    abs_relu6_kernel<<<blocks, block_size, 0, stream>>>(input, output, numel);
}
