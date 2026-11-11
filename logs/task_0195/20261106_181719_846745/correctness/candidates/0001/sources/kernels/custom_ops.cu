#include <cuda_runtime.h>

// Elementwise ReLU kernel
__global__ void relu_kernel(float* data, long long numel) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < numel; i += stride) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

// Launcher for fused Conv+ReLU, called from C++ binding after cuDNN convolution.
// The binding is responsible for performing the convolution with cuDNN and
// then calling this function to apply ReLU on the output.
extern "C" void conv2d_relu_launcher(
    float* conv_output,
    long long numel,
    cudaStream_t stream) {
    int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    if (blocks_ll > 0) {
        int blocks = (int)blocks_ll;
        relu_kernel<<<blocks, threads, 0, stream>>>(conv_output, numel);
    }
}
