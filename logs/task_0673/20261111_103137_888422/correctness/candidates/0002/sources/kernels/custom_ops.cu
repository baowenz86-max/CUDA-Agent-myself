#include <cuda_runtime.h>

// Kernel for concatenating two tensors of shape [N, C, H, W] along channel dimension (dim=1)
// x1 shape [N, C1, H, W], x2 shape [N, C2, H, W], output shape [N, C1+C2, H, W]
// All tensors are contiguous float32 arrays. total_elements = N * (C1+C2) * H * W
__global__ void cat_dim1_kernel(float* __restrict__ out, const float* __restrict__ x1, const float* __restrict__ x2,
                                long long N, long long C1, long long C2, long long H, long long W,
                                long long total_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    long long HW = H * W;
    long long C1HW = C1 * HW;
    long long C2HW = C2 * HW;
    long long CoutHW = (C1 + C2) * HW;
    for (long long i = idx; i < total_elements; i += stride) {
        long long n = i / CoutHW;
        long long rem = i % CoutHW;
        long long c = rem / HW;
        long long hw = rem % HW;
        if (c < C1) {
            out[i] = x1[n * C1HW + c * HW + hw];
        } else {
            long long c2 = c - C1;
            out[i] = x2[n * C2HW + c2 * HW + hw];
        }
    }
}

// Kernel for element-wise sigmoid: 1/(1+exp(-x))
__global__ void sigmoid_kernel(const float* __restrict__ x, float* __restrict__ y, long long total_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (long long i = idx; i < total_elements; i += stride) {
        y[i] = 1.0f / (1.0f + __expf(-x[i]));
    }
}

extern "C" void cat_launcher(float* output, const float* x1, const float* x2, long long N, long long C1, long long C2, long long H, long long W, cudaStream_t stream) {
    long long total_elements = N * (C1 + C2) * H * W;
    const int block_size = 256;
    int grid_size = (int)std::min((long long)65535, (total_elements + block_size - 1) / block_size);
    if (grid_size == 0) grid_size = 1;
    cat_dim1_kernel<<<grid_size, block_size, 0, stream>>>(output, x1, x2, N, C1, C2, H, W, total_elements);
}

extern "C" void sigmoid_launcher(const float* input, float* output, long long total_elements, cudaStream_t stream) {
    const int block_size = 256;
    int grid_size = (int)std::min((long long)65535, (total_elements + block_size - 1) / block_size);
    if (grid_size == 0) grid_size = 1;
    sigmoid_kernel<<<grid_size, block_size, 0, stream>>>(input, output, total_elements);
}
