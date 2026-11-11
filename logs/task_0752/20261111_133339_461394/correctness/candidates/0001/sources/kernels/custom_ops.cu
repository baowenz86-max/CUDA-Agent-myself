#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Fused concatenation along dim=1 followed by erf.
// Inputs: two tensors with same shape except dim=1.
// in0: [N, C0, H, W], in1: [N, C1, H, W]
// output: [N, C0+C1, H, W] = erf(cat(in0, in1, dim=1))
// All tensors are contiguous (reference inputs are contiguous, output allocated contiguous).
// Boundary checks included for safety.

template <typename T>
__global__ void cat_erf_kernel(
    T* __restrict__ out,
    const T* __restrict__ in0,
    const T* __restrict__ in1,
    int N, int C0, int C1, int H, int W)
{
    int total = N * (C0 + C1) * H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        // decompose i -> n, c, hw
        int hw = H * W;
        int plane = i / hw;
        int hw_idx = i - plane * hw;
        int n = plane / (C0 + C1);
        int c = plane - n * (C0 + C1);

        const T* src;
        int src_idx;
        if (c < C0) {
            src = in0;
            src_idx = (n * C0 + c) * hw + hw_idx;
        } else {
            src = in1;
            int c1 = c - C0;
            src_idx = (n * C1 + c1) * hw + hw_idx;
        }
        out[i] = erff(src[src_idx]);
    }
}

// Launcher: assumes out, in0, in1 are valid CUDA float pointers.
// All sizes > 0. If any size is zero, does nothing (not expected in this task).
extern "C" void custom_cat_erf_launcher(
    float* out,
    const float* in0,
    const float* in1,
    int N, int C0, int C1, int H, int W,
    cudaStream_t stream)
{
    if (N <= 0 || C0 <= 0 || C1 <= 0 || H <= 0 || W <= 0) return;

    int total = N * (C0 + C1) * H * W;
    const int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    // Cap grid to avoid too many blocks for very large tensors, use grid-stride loop.
    // For total ~209M, grid_size would be ~819k; cap to 65535.
    if (grid_size > 65535) grid_size = 65535;

    cat_erf_kernel<float><<<grid_size, block_size, 0, stream>>>(
        out, in0, in1, N, C0, C1, H, W);
}
