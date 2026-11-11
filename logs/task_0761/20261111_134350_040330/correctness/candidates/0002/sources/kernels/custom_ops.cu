#include <cuda_runtime.h>

// Helper to compute ceiling division
// Make it available on both host and device
template <typename T>
__host__ __device__ __forceinline__ T ceil_div(T a, T b) {
    return (a + b - 1) / b;
}

// Softmax along dim=1 for a 3D tensor of shape [B, D, C]
// Input and output are contiguous float32 arrays.
// Each (b, d, c) element is indexed as ((b * D + d) * C + c)
__global__ void softmax1d_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 int B, int D, int C) {
    int total = B * D * C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int linear = idx; linear < total; linear += stride) {
        int c = linear % C;
        int tmp = linear / C;
        int d = tmp % D;
        int b = tmp / D;

        // Compute max over d' for this (b, c)
        float max_val = -INFINITY;
        for (int dd = 0; dd < D; ++dd) {
            float val = input[((b * D + dd) * C + c)];
            if (val > max_val) max_val = val;
        }

        // Compute sum of exp(x - max) over d'
        float sum = 0.0f;
        for (int dd = 0; dd < D; ++dd) {
            float val = input[((b * D + dd) * C + c)];
            sum += __expf(val - max_val);
        }

        float x_val = input[((b * D + d) * C + c)];
        output[linear] = __expf(x_val - max_val) / sum;
    }
}

// Simple elementwise copy kernel
template <typename T>
__global__ void copy_kernel(const T* __restrict__ input, T* __restrict__ output, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        output[i] = input[i];
    }
}

// Elementwise add scalar kernel
template <typename T, typename Scalar>
__global__ void add_scalar_kernel(const T* __restrict__ input, T* __restrict__ output, Scalar value, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        output[i] = input[i] + static_cast<T>(value);
    }
}

// Launchers (extern "C" to be callable from C++ binding)
extern "C" void softmax1d_launcher(const float* input, float* output,
                                    int B, int D, int C, cudaStream_t stream) {
    int total = B * D * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Grid-stride loop handles large arrays; we could also use a fixed grid.
    // Cap blocks to avoid exceeding grid dimension limits (optional).
    int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    softmax1d_kernel<<<blocks, threads, 0, stream>>>(input, output, B, D, C);
}

extern "C" void squeeze_launcher(const float* input, float* output, size_t numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)ceil_div<size_t>(numel, (size_t)threads);
    if (blocks > 65535) blocks = 65535;
    copy_kernel<float><<<blocks, threads, 0, stream>>>(input, output, numel);
}

extern "C" void add_scalar_launcher(const float* input, float* output, double value, size_t numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)ceil_div<size_t>(numel, (size_t)threads);
    if (blocks > 65535) blocks = 65535;
    add_scalar_kernel<float, double><<<blocks, threads, 0, stream>>>(input, output, value, numel);
}
