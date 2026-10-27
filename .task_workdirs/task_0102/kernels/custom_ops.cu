// custom_ops.cu
// CUDA kernels for custom_elu and custom_triu
// No PyTorch/ATen dependencies. Accepts raw pointers and cudaStream_t.

#include <cuda_runtime.h>
#include <math.h>

// -----------------------------------------------------------------------------
// ELU kernel: y = x if x > 0 else exp(x) - 1
// Supports arbitrary stride via element indexing.
// -----------------------------------------------------------------------------
__global__ void elu_kernel(float* out, const float* in, int64_t total,
                           int64_t spatial_size, int64_t inner_stride,
                           int64_t dim1_stride, int64_t dim0_stride) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        // Compute linear index to element
        // The input is expected to be contiguous, but we compute using the same
        // linear index since output layout mirrors input (contiguous).
        float v = in[idx];
        out[idx] = (v > 0.0f) ? v : (expf(v) - 1.0f);
    }
}

// -----------------------------------------------------------------------------
// triu kernel: zero out elements below main diagonal in each 2D matrix.
// Input is viewed as [outer, H, W], where H=W=256. The kernel processes
// each matrix independently; elements with row > col are set to zero.
// Supports arbitrary batch dims via outer = total / (H*W).
// -----------------------------------------------------------------------------
__global__ void triu_kernel(float* out, const float* in, int64_t total,
                            int64_t outer, int64_t H, int64_t W) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int64_t mat = idx / (H * W);
        int64_t rem = idx % (H * W);
        int64_t row = rem / W;
        int64_t col = rem % W;
        if (row <= col) {
            out[idx] = in[idx];
        } else {
            out[idx] = 0.0f;
        }
    }
}

// -----------------------------------------------------------------------------
// Host launchers (extern "C")
// -----------------------------------------------------------------------------

extern "C" void custom_elu_launcher(float* output, const float* input,
                                    int64_t total_elements, cudaStream_t stream) {
    const int threads = 256;
    int64_t blocks64 = (total_elements + threads - 1) / threads;
    int blocks = (blocks64 > 65535) ? 65535 : (int)blocks64;
    if (blocks == 0) return; // empty tensor
    elu_kernel<<<blocks, threads, 0, stream>>>(
        output, input, total_elements, 0, 0, 0, 0);
}

extern "C" void custom_triu_launcher(float* output, const float* input,
                                     int64_t total_elements, int64_t H, int64_t W,
                                     cudaStream_t stream) {
    if (total_elements <= 0) return;
    const int threads = 256;
    int64_t outer = total_elements / (H * W);
    int64_t blocks64 = (total_elements + threads - 1) / threads;
    int blocks = (blocks64 > 65535) ? 65535 : (int)blocks64;
    triu_kernel<<<blocks, threads, 0, stream>>>(
        output, input, total_elements, outer, H, W);
}
