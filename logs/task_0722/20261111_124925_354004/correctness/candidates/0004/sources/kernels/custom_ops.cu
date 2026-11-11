#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Utility: div_up
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__ int div_up(int a, int b) {
    return (a + b - 1) / b;
}

// -----------------------------------------------------------------------------
// Kernel: custom_mean_impl
// Computes mean over dimension dim (0-indexed) for a 4D tensor of shape (N,C,H,W)
// Requires keepdim = true, output shape (N,C,1,W)
// Uses one thread per output element; each thread loops over reduction dim H.
// Handles arbitrary strides through explicit index computation.
// Uses double accumulation for improved numerical accuracy matching PyTorch.
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void custom_mean_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int N, int C, int H, int W,
    int in_stride_N, int in_stride_C, int in_stride_H, int in_stride_W,
    int out_stride_N, int out_stride_C, int out_stride_W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * C * W;  // since keepdim true, output shape N,C,1,W
    if (idx >= total_outputs) return;

    // Decompose idx into (n, c, w)
    int w = idx % W;
    int tmp = idx / W;
    int c = tmp % C;
    int n = tmp / C;

    const scalar_t* base = input + n * in_stride_N + c * in_stride_C + w * in_stride_W;
    double sum = 0.0;
    for (int h = 0; h < H; ++h) {
        sum += static_cast<double>(base[h * in_stride_H]);
    }
    output[n * out_stride_N + c * out_stride_C + w * out_stride_W] = static_cast<scalar_t>(sum / H);
}

// -----------------------------------------------------------------------------
// Launcher: custom_mean_launcher
// Computes mean along dim for 4D float32 tensor, keepdim always true.
// -----------------------------------------------------------------------------
extern "C" void custom_mean_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int in_stride_N, int in_stride_C, int in_stride_H, int in_stride_W,
    int out_stride_N, int out_stride_C, int out_stride_W,
    cudaStream_t stream
) {
    int total_outputs = N * C * W;
    int block = 256;
    int grid = div_up(total_outputs, block);
    custom_mean_kernel<float><<<grid, block, 0, stream>>>(
        input, output,
        N, C, H, W,
        in_stride_N, in_stride_C, in_stride_H, in_stride_W,
        out_stride_N, out_stride_C, out_stride_W
    );
}

// -----------------------------------------------------------------------------
// Kernel: custom_gt_impl
// Computes input > other with broadcasting along dimension 2 (height).
// Input shape (N,C,H,W), other shape (N,C,1,W), output bool shape (N,C,H,W).
// One thread per output element; handles arbitrary strides.
// -----------------------------------------------------------------------------
template <typename scalar_t>
__global__ void custom_gt_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ other,
    bool* __restrict__ output,
    int N, int C, int H, int W,
    int in_stride_N, int in_stride_C, int in_stride_H, int in_stride_W,
    int other_stride_N, int other_stride_C, int other_stride_W,
    int out_stride_N, int out_stride_C, int out_stride_H, int out_stride_W
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;
    if (idx >= total_elements) return;

    int w = idx % W;
    int tmp = idx / W;
    int h = tmp % H;
    tmp = tmp / H;
    int c = tmp % C;
    int n = tmp / C;

    scalar_t input_val = input[n * in_stride_N + c * in_stride_C + h * in_stride_H + w * in_stride_W];
    scalar_t other_val = other[n * other_stride_N + c * other_stride_C + w * other_stride_W];
    output[n * out_stride_N + c * out_stride_C + h * out_stride_H + w * out_stride_W] =
        (input_val > other_val);
}

// -----------------------------------------------------------------------------
// Launcher: custom_gt_launcher
// Performs broadcasting greater-than for float32 tensors.
// -----------------------------------------------------------------------------
extern "C" void custom_gt_launcher(
    const float* input,
    const float* other,
    bool* output,
    int N, int C, int H, int W,
    int in_stride_N, int in_stride_C, int in_stride_H, int in_stride_W,
    int other_stride_N, int other_stride_C, int other_stride_W,
    int out_stride_N, int out_stride_C, int out_stride_H, int out_stride_W,
    cudaStream_t stream
) {
    int total_elements = N * C * H * W;
    int block = 256;
    int grid = div_up(total_elements, block);
    custom_gt_kernel<float><<<grid, block, 0, stream>>>(
        input, other, output,
        N, C, H, W,
        in_stride_N, in_stride_C, in_stride_H, in_stride_W,
        other_stride_N, other_stride_C, other_stride_W,
        out_stride_N, out_stride_C, out_stride_H, out_stride_W
    );
}
