#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// Kernel for custom_max_dim1: computes max over dimension 1.
// Input x has shape [N, C, H, W], output has shape [N, H, W].
// x_stride[0..3] are strides in elements for each dimension.
// out is contiguous with strides [H*W, W, 1].
__global__ void max_dim1_kernel(const float* __restrict__ x,
                                float* __restrict__ out,
                                int N, int C, int H, int W,
                                long long x_stride0, long long x_stride1,
                                long long x_stride2, long long x_stride3) {
    long long total = (long long)N * H * W;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int w = idx % W;
        long long tmp = idx / W;
        int h = tmp % H;
        int n = tmp / H;

        const float* base = x + (long long)n * x_stride0 + (long long)h * x_stride2 + (long long)w * x_stride3;
        float max_val = -INFINITY;
        for (int c = 0; c < C; ++c) {
            float val = base[c * x_stride1];
            if (val > max_val) max_val = val;
        }
        out[idx] = max_val;
    }
}

// Kernel for custom_exp: element-wise expf.
__global__ void exp_kernel(const float* __restrict__ x,
                           float* __restrict__ out,
                           long long numel,
                           long long stride) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long grid_stride = (long long)blockDim.x * gridDim.x;
    for (; idx < numel; idx += grid_stride) {
        out[idx] = expf(x[idx * stride]);
    }
}

// Launcher for custom_max_dim1.
// x_data: pointer to input tensor data (float32)
// out_data: pointer to output tensor data (float32), already allocated contiguous
// N, C, H, W: dimensions
// strides: array of 4 strides (in elements) for input tensor
// stream: CUDA stream to launch kernel on
extern "C" void custom_max_dim1_launcher(const float* x_data,
                                         float* out_data,
                                         int N, int C, int H, int W,
                                         const long long* strides,
                                         cudaStream_t stream) {
    long long total = (long long)N * H * W;
    const int threads = 256;
    long long blocks_ll = (total + threads - 1) / threads;
    int blocks = (blocks_ll > 2147483647LL) ? 2147483647 : (int)blocks_ll;

    max_dim1_kernel<<<blocks, threads, 0, stream>>>(
        x_data, out_data, N, C, H, W,
        strides[0], strides[1], strides[2], strides[3]);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "custom_max_dim1 kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

// Launcher for custom_exp.
// x_data: pointer to input tensor data (float32)
// out_data: pointer to output tensor data (float32), already allocated with same shape as input
// numel: number of elements
// stride: stride in elements between consecutive elements in x (for non-contiguous, may be >1; for contiguous, =1)
// stream: CUDA stream
extern "C" void custom_exp_launcher(const float* x_data,
                                    float* out_data,
                                    long long numel,
                                    long long stride,
                                    cudaStream_t stream) {
    const int threads = 256;
    long long blocks_ll = (numel + threads - 1) / threads;
    int blocks = (blocks_ll > 2147483647LL) ? 2147483647 : (int)blocks_ll;

    exp_kernel<<<blocks, threads, 0, stream>>>(x_data, out_data, numel, stride);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "custom_exp kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}
