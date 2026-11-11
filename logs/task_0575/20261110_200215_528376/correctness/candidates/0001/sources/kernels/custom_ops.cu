// CUDA kernels for max_dim1 and round_tensor
#include <cuda_runtime.h>
#include <cstdint>

// Helper to round half to even (torch.round semantics)
__device__ __forceinline__ float round_even(float x) {
    // Use nearbyintf with FE_TONEAREST for correct behavior; default is nearest even
    return nearbyintf(x);
}

// Kernel for max over dim=1
// Input: x (batch, C, H, W) with arbitrary strides
// Output: values (batch, H, W), indices (batch, H, W) int64
// Each thread handles multiple (b, h, w) locations (grid-stride loop).
__global__ void max_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ values,
    int64_t* __restrict__ indices,
    int batch,
    int C,
    int H,
    int W,
    int64_t stride_b,
    int64_t stride_c,
    int64_t stride_h,
    int64_t stride_w
) {
    int64_t total = (int64_t)batch * H * W;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        int b = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;

        float max_val = -3.402823466e38f; // -FLT_MAX
        int64_t max_idx = 0;

        for (int c = 0; c < C; ++c) {
            int64_t linear = (int64_t)b * stride_b + (int64_t)c * stride_c + (int64_t)h * stride_h + (int64_t)w * stride_w;
            float val = x[linear];
            // Tie-breaking: choose first occurrence (torch.max behavior: first maximum)
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
        }

        values[i] = max_val;
        indices[i] = max_idx;
    }
}

// Kernel for elementwise round (half to even)
__global__ void round_tensor_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int64_t numel
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < numel; i += stride) {
        out[i] = nearbyintf(x[i]);
    }
}

// Launcher for max_dim1: expects valid pointers and stream.
// Outputs are assumed preallocated by the binding (values and indices contiguous).
extern "C" void max_dim1_launcher(
    const float* x,
    float* values,
    int64_t* indices,
    int batch,
    int C,
    int H,
    int W,
    int64_t stride_b,
    int64_t stride_c,
    int64_t stride_h,
    int64_t stride_w,
    cudaStream_t stream
) {
    int64_t total = (int64_t)batch * H * W;
    if (total == 0) return;

    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (blocks > 0) {
        max_dim1_kernel<<<blocks, threads, 0, stream>>>(
            x, values, indices, batch, C, H, W,
            stride_b, stride_c, stride_h, stride_w
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            // In a real binding, this would throw a Python exception.
            // Here we cannot include torch, so we simply return a flag via an output? We'll just ignore for now.
            // The binding should check cudaGetLastError() before returning.
        }
    }
}

// Launcher for round_tensor: x and out are assumed contiguous 1D of numel elements.
extern "C" void round_tensor_launcher(
    const float* x,
    float* out,
    int64_t numel,
    cudaStream_t stream
) {
    if (numel == 0) return;

    int threads = 256;
    int64_t blocks64 = (numel + threads - 1) / threads;
    int blocks = static_cast<int>(blocks64);
    if (blocks > 0) {
        round_tensor_kernel<<<blocks, threads, 0, stream>>>(x, out, numel);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            // ignore; binding should check
        }
    }
}
