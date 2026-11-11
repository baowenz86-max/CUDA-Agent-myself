#include <cuda_runtime.h>
#include <stdint.h>

// mean_reduce_kernel: 每个线程处理一个 batch 元素，使用 double 累计提高精度
__global__ void mean_reduce_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int64_t batch,
                                   int64_t inner_size,
                                   int64_t batch_stride) {
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch) return;
    const float* ptr = input + b * batch_stride;
    double sum = 0.0;
    for (int64_t i = 0; i < inner_size; ++i) {
        sum += static_cast<double>(ptr[i]);
    }
    output[b] = static_cast<float>(sum / static_cast<double>(inner_size));
}

// remainder_kernel: 逐元素求余，使用 double 计算后转为 float
__global__ void remainder_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int64_t numel,
                                 double divisor) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    double x = static_cast<double>(input[idx]);
    double result = fmod(x, divisor);
    output[idx] = static_cast<float>(result);
}

// launcher for mean_reduce_op
extern "C" void mean_reduce_op_launcher(const float* input,
                                        float* output,
                                        int64_t batch,
                                        int64_t inner_size,
                                        int64_t batch_stride,
                                        cudaStream_t stream) {
    int threads = 256;
    int blocks = (batch + threads - 1) / threads;
    mean_reduce_kernel<<<blocks, threads, 0, stream>>>(input, output, batch, inner_size, batch_stride);
}

// launcher for remainder_op
extern "C" void remainder_op_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      double divisor,
                                      cudaStream_t stream) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    remainder_kernel<<<blocks, threads, 0, stream>>>(input, output, numel, divisor);
}
