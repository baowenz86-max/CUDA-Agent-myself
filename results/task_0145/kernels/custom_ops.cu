#include <cuda_runtime.h>

// Fused kernel: applies lower-triangular mask and instance normalization (eval mode)
// Input x: shape [N, C, H, W] with arbitrary strides, float32, CUDA
// Output: same shape as x, but contiguous (float32)
// running_mean, running_var: shape [C], contiguous float32
// eps: float scalar
// diagonal: integer diagonal offset (0 for main diagonal)
// The kernel computes:
//   if (h >= w + diagonal) {  // element is in lower triangle (or on diagonal when diagonal=0)
//       c = channel index
//       output = (input - running_mean[c]) * rsqrtf(running_var[c] + eps);
//   } else {
//       output = 0;
//   }
__global__ void fused_tril_instance_norm_eval_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int diagonal,
    int total_elements,
    int C,
    int H,
    int W,
    int stride_N,
    int stride_C,
    int stride_H,
    int stride_W)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        // Decompose linear index into (n, c, h, w)
        int n = i / (C * H * W);
        int rem = i % (C * H * W);
        int c = rem / (H * W);
        int rem2 = rem % (H * W);
        int h = rem2 / W;
        int w = rem2 % W;

        // Check lower triangular condition
        bool in_tril = (h >= (w + diagonal));
        if (in_tril) {
            int input_offset = n * stride_N + c * stride_C + h * stride_H + w * stride_W;
            float val = input[input_offset] - running_mean[c];
            float inv_std = rsqrtf(running_var[c] + eps);
            output[i] = val * inv_std;
        } else {
            output[i] = 0.0f;
        }
    }
}

extern "C" void fused_tril_instance_norm_eval_forward(
    const float* input,
    float* output,
    const float* running_mean,
    const float* running_var,
    float eps,
    int diagonal,
    int N,
    int C,
    int H,
    int W,
    int stride_N,
    int stride_C,
    int stride_H,
    int stride_W,
    cudaStream_t stream)
{
    int total_elements = N * C * H * W;

    // Choose block size and grid size
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    // Cap grid to avoid launching too many blocks (though total_elements is large)
    if (grid_size > 65535) {
        grid_size = 65535;
    }

    fused_tril_instance_norm_eval_kernel<<<grid_size, block_size, 0, stream>>>(
        input,
        output,
        running_mean,
        running_var,
        eps,
        diagonal,
        total_elements,
        C,
        H,
        W,
        stride_N,
        stride_C,
        stride_H,
        stride_W);

    // Error checking (optional but good practice)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // In a real binding layer, we would throw an exception; here we just ignore.
        // The launcher is designed to be called from a binding that checks stream errors later.
    }
}
