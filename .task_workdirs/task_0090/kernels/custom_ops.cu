#include <cuda_runtime.h>

// 逐元素 ELU kernel: y = x if x > 0 else expf(x) - 1.f
__global__ void elu_kernel(const float* __restrict__ input,
                           float* __restrict__ output,
                           long long n) {
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (; idx < n; idx += stride) {
        float x = input[idx];
        output[idx] = x > 0.f ? x : expf(x) - 1.f;
    }
}

// 逐元素 sigmoid kernel: y = 1 / (1 + exp(-x))
__global__ void sigmoid_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               long long n) {
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (; idx < n; idx += stride) {
        float x = input[idx];
        output[idx] = 1.f / (1.f + expf(-x));
    }
}

// 融合 kernel: 先 ELU，再 sigmoid
__global__ void elu_sigmoid_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   long long n) {
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (; idx < n; idx += stride) {
        float x = input[idx];
        float elu = x > 0.f ? x : expf(x) - 1.f;
        output[idx] = 1.f / (1.f + expf(-elu));
    }
}

// ============ Launchers (extern "C" for binding) ============

extern "C" void elu_forward_launcher(const float* input,
                                     float* output,
                                     long long n,
                                     cudaStream_t stream) {
    int block_size = 256;
    long long blocks_ll = (n + block_size - 1) / block_size;
    int blocks = static_cast<int>(min(blocks_ll, 1073741824LL));
    elu_kernel<<<blocks, block_size, 0, stream>>>(input, output, n);
    cudaGetLastError();
}

extern "C" void sigmoid_forward_launcher(const float* input,
                                          float* output,
                                          long long n,
                                          cudaStream_t stream) {
    int block_size = 256;
    long long blocks_ll = (n + block_size - 1) / block_size;
    int blocks = static_cast<int>(min(blocks_ll, 1073741824LL));
    sigmoid_kernel<<<blocks, block_size, 0, stream>>>(input, output, n);
    cudaGetLastError();
}

extern "C" void elu_sigmoid_forward_launcher(const float* input,
                                              float* output,
                                              long long n,
                                              cudaStream_t stream) {
    int block_size = 256;
    long long blocks_ll = (n + block_size - 1) / block_size;
    int blocks = static_cast<int>(min(blocks_ll, 1073741824LL));
    elu_sigmoid_kernel<<<blocks, block_size, 0, stream>>>(input, output, n);
    cudaGetLastError();
}
