#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <cstdio>

// Helper macro for CUDA error checking
template <typename T>
void check(T result, const char* const func, const char* const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// Kernel: compute max along dim=1 (C dimension) for each (n,h,w)
// Input: contiguous data (after possible copy), shape N x C x H x W
// Output: max_val shape N x 1 x H x W
template <int BLOCK_SIZE>
__global__ void max_dim1_kernel(const float* __restrict__ input,
                                float* __restrict__ output,
                                int N, int C, int H, int W) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = N * H * W;
    if (idx >= total) return;

    int n = idx / (H * W);
    int rem = idx % (H * W);
    int h = rem / W;
    int w = rem % W;

    const float* base = input + n * C * H * W + h * W + w;
    float max_val = -1e30f; // very small value
    for (int c = 0; c < C; ++c) {
        float val = base[c * H * W];
        if (val > max_val) max_val = val;
    }
    output[idx] = max_val;
}

// Kernel: compute argmax (first occurrence of max) along dim=1
template <int BLOCK_SIZE>
__global__ void argmax_dim1_kernel(const float* __restrict__ input,
                                   int64_t* __restrict__ output,
                                   int N, int C, int H, int W) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = N * H * W;
    if (idx >= total) return;

    int n = idx / (H * W);
    int rem = idx % (H * W);
    int h = rem / W;
    int w = rem % W;

    const float* base = input + n * C * H * W + h * W + w;
    float max_val = -1e30f;
    int max_idx = 0;
    for (int c = 0; c < C; ++c) {
        float val = base[c * H * W];
        if (val > max_val) {
            max_val = val;
            max_idx = c;
        }
    }
    output[idx] = static_cast<int64_t>(max_idx);
}

// Kernel: multiply each element of input by scalar (scalar is a single float)
template <int BLOCK_SIZE>
__global__ void scalar_mul_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  float scalar,
                                  int total) {
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx >= total) return;
    output[idx] = input[idx] * scalar;
}

// Global copy kernel definition
__global__ void copy_contiguous_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                       int N, int C, int H, int W,
                                       int64_t sN, int64_t sC, int64_t sH, int64_t sW,
                                       int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int n = idx / (C * H * W);
    int rem = idx % (C * H * W);
    int c = rem / (H * W);
    int rem2 = rem % (H * W);
    int h = rem2 / W;
    int w = rem2 % W;
    dst[idx] = src[n * sN + c * sC + h * sH + w * sW];
}

// Helper function to copy non-contiguous 4D tensor to contiguous memory
// input: pointer to original data (may be non-contiguous)
// shape: N,C,H,W, strides: sN,sC,sH,sW
// Returns contiguous array allocated on device (caller must free)
float* make_contiguous(const float* input,
                       int N, int C, int H, int W,
                       int64_t sN, int64_t sC, int64_t sH, int64_t sW,
                       cudaStream_t stream) {
    size_t size = static_cast<size_t>(N) * C * H * W * sizeof(float);
    float* contig;
    checkCudaErrors(cudaMalloc(&contig, size));
    // Launch kernel to copy element-wise using strides
    int total = N * C * H * W;
    int block = 256;
    int grid = (total + block - 1) / block;
    copy_contiguous_kernel<<<grid, block, 0, stream>>>(input, contig, N, C, H, W, sN, sC, sH, sW, total);
    checkCudaErrors(cudaGetLastError());
    return contig;
}

// Launchers exposed to C++ (binding)
extern "C" {

void custom_max_dim1_launcher(const float* input,
                              float* output,
                              int N, int C, int H, int W,
                              int64_t sN, int64_t sC, int64_t sH, int64_t sW,
                              cudaStream_t stream) {
    // Check if input is contiguous (strides match contiguous layout)
    bool contiguous = (sC == (int64_t)H * W && sH == W && sW == 1 && sN == (int64_t)C * H * W);
    const float* data = input;
    float* temp = nullptr;
    if (!contiguous) {
        temp = make_contiguous(input, N, C, H, W, sN, sC, sH, sW, stream);
        data = temp;
    }

    int total = N * H * W;
    const int block = 256;
    int grid = (total + block - 1) / block;
    max_dim1_kernel<block><<<grid, block, 0, stream>>>(data, output, N, C, H, W);
    checkCudaErrors(cudaGetLastError());

    if (temp) {
        checkCudaErrors(cudaFree(temp));
    }
}

void custom_argmax_dim1_launcher(const float* input,
                                 int64_t* output,
                                 int N, int C, int H, int W,
                                 int64_t sN, int64_t sC, int64_t sH, int64_t sW,
                                 cudaStream_t stream) {
    bool contiguous = (sC == (int64_t)H * W && sH == W && sW == 1 && sN == (int64_t)C * H * W);
    const float* data = input;
    float* temp = nullptr;
    if (!contiguous) {
        temp = make_contiguous(input, N, C, H, W, sN, sC, sH, sW, stream);
        data = temp;
    }

    int total = N * H * W;
    const int block = 256;
    int grid = (total + block - 1) / block;
    argmax_dim1_kernel<block><<<grid, block, 0, stream>>>(data, output, N, C, H, W);
    checkCudaErrors(cudaGetLastError());

    if (temp) {
        checkCudaErrors(cudaFree(temp));
    }
}

void custom_scalar_mul_launcher(const float* input,
                                float* output,
                                float scalar_value,
                                int total,
                                cudaStream_t stream) {
    const int block = 256;
    int grid = (total + block - 1) / block;
    scalar_mul_kernel<block><<<grid, block, 0, stream>>>(input, output, scalar_value, total);
    checkCudaErrors(cudaGetLastError());
}

} // extern "C"
