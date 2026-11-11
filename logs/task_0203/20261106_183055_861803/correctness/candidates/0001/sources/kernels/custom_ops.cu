#include <cuda_runtime.h>
#include <cmath>

// GELU activation: 0.5 * x * (1 + erf(x / sqrt(2)))
__global__ void gelu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            long long total_elements,
                            int rank,
                            const long long* __restrict__ input_strides,
                            const long long* __restrict__ output_strides,
                            const long long* __restrict__ shape) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long linear = idx; linear < total_elements; linear += stride) {
        // 将线性索引映射到多维坐标，考虑任意 stride
        long long remaining = linear;
        long long input_offset = 0;
        long long output_offset = 0;
        for (int d = 0; d < rank; ++d) {
            long long coord = remaining % shape[d];
            remaining /= shape[d];
            input_offset += coord * input_strides[d];
            output_offset += coord * output_strides[d];
        }

        float x = input[input_offset];
        float gelu_val = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
        output[output_offset] = gelu_val;
    }
}

// Truncation toward zero
__global__ void trunc_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             long long total_elements,
                             int rank,
                             const long long* __restrict__ input_strides,
                             const long long* __restrict__ output_strides,
                             const long long* __restrict__ shape) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long linear = idx; linear < total_elements; linear += stride) {
        long long remaining = linear;
        long long input_offset = 0;
        long long output_offset = 0;
        for (int d = 0; d < rank; ++d) {
            long long coord = remaining % shape[d];
            remaining /= shape[d];
            input_offset += coord * input_strides[d];
            output_offset += coord * output_strides[d];
        }

        float x = input[input_offset];
        output[output_offset] = truncf(x);
    }
}

// Helper: 将连续输出步幅写入 output_strides
__global__ void compute_contiguous_strides_kernel(long long* output_strides,
                                                  const long long* shape,
                                                  int rank) {
    // 只需要一个线程执行
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    long long stride = 1;
    for (int d = rank - 1; d >= 0; --d) {
        output_strides[d] = stride;
        stride *= shape[d];
    }
}

extern "C" void gelu_forward_launcher(const float* input,
                                       float* output,
                                       const long long* shape,
                                       const long long* input_strides,
                                       int rank,
                                       long long total_elements,
                                       cudaStream_t stream) {
    // 先计算连续输出 strides
    long long* d_output_strides;
    cudaMalloc(&d_output_strides, rank * sizeof(long long));
    cudaMemsetAsync(d_output_strides, 0, rank * sizeof(long long), stream);

    cudaMemcpyAsync(d_output_strides, shape, rank * sizeof(long long), cudaMemcpyDeviceToDevice, stream); // 临时占位，实际在 kernel 中计算
    // 更安全：直接调用 kernel 计算 strides，再传入 gelu kernel
    compute_contiguous_strides_kernel<<<1, 1, 0, stream>>>(d_output_strides, shape, rank);

    int threads = 256;
    long long blocks_ll = (total_elements + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL; // 32位 int 上限
    int blocks = (int)blocks_ll;

    gelu_kernel<<<blocks, threads, 0, stream>>>(
        input, output, total_elements, rank, input_strides, d_output_strides, shape);

    cudaFree(d_output_strides);
}

extern "C" void trunc_forward_launcher(const float* input,
                                       float* output,
                                       const long long* shape,
                                       const long long* input_strides,
                                       int rank,
                                       long long total_elements,
                                       cudaStream_t stream) {
    long long* d_output_strides;
    cudaMalloc(&d_output_strides, rank * sizeof(long long));
    cudaMemsetAsync(d_output_strides, 0, rank * sizeof(long long), stream);

    cudaMemcpyAsync(d_output_strides, shape, rank * sizeof(long long), cudaMemcpyDeviceToDevice, stream);
    compute_contiguous_strides_kernel<<<1, 1, 0, stream>>>(d_output_strides, shape, rank);

    int threads = 256;
    long long blocks_ll = (total_elements + threads - 1) / threads;
    if (blocks_ll > 2147483647LL) blocks_ll = 2147483647LL;
    int blocks = (int)blocks_ll;

    trunc_kernel<<<blocks, threads, 0, stream>>>(
        input, output, total_elements, rank, input_strides, d_output_strides, shape);

    cudaFree(d_output_strides);
}
