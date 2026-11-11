#include <cuda_runtime.h>
#include <cuComplex.h>

// Custom CUDA kernels and launchers for the model.
// No torch/ATen dependencies.

// Copy kernel: copies numel float elements from host to device.
__global__ void copy_to_cuda_kernel(float* __restrict__ d_out,
                                    const float* __restrict__ h_in,
                                    size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < numel; i += stride) {
        d_out[i] = h_in[i];
    }
}

// Max over dim=1 kernel.
// Input layout: [N, 2, H, W, 2] (float)
// Output values: [N, 1, H, W, 2] (float)
// Output indices: [N, 1, H, W, 2] (int64_t)
// Each thread handles one (n, h, w, last_dim) position.
__global__ void max_step_kernel(const float* __restrict__ input,
                                float* __restrict__ values,
                                int64_t* __restrict__ indices,
                                int N, int H, int W) {
    // Total number of output elements per last_dim? We'll use flattened index.
    // Output shape [N,1,H,W,2] -> total = N*H*W*2.
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t out_stride = blockDim.x * gridDim.x;
    size_t total_out = (size_t)N * H * W * 2;
    for (size_t idx = out_idx; idx < total_out; idx += out_stride) {
        // Decode idx: idx = (((n*H + h)*W + w)*2 + last)
        int last = idx % 2;
        size_t tmp = idx / 2;
        int w = tmp % W;
        tmp /= W;
        int h = tmp % H;
        int n = tmp / H;
        // Base offset for channel 0 and channel 1.
        size_t base0 = (((size_t)n * 2 + 0) * H + h) * W * 2 + (size_t)w * 2 + last;
        size_t base1 = (((size_t)n * 2 + 1) * H + h) * W * 2 + (size_t)w * 2 + last;
        float v0 = input[base0];
        float v1 = input[base1];
        // PyTorch max semantics: if equal, return first index (0).
        float max_val = v0;
        int64_t max_idx = 0;
        if (v1 > v0) {
            max_val = v1;
            max_idx = 1;
        }
        values[idx] = max_val;
        indices[idx] = max_idx;
    }
}

// View as complex kernel: copies float2 pairs to complex64.
// Input: [N, H, W, 2] float contiguous.
// Output: [N, H, W] complex64 (cuComplex = float2).
__global__ void view_as_complex_kernel(const float* __restrict__ input,
                                       cuComplex* __restrict__ output,
                                       int N, int H, int W) {
    size_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t out_stride = blockDim.x * gridDim.x;
    size_t total_out = (size_t)N * H * W;
    for (size_t idx = out_idx; idx < total_out; idx += out_stride) {
        // input index = idx * 2
        float real = input[idx * 2];
        float imag = input[idx * 2 + 1];
        output[idx] = make_cuComplex(real, imag);
    }
}

// Launchers
extern "C" void copy_to_cuda_launcher(float* d_out,
                                      const float* h_in,
                                      size_t numel,
                                      cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    copy_to_cuda_kernel<<<blocks, threads, 0, stream>>>(d_out, h_in, numel);
}

extern "C" void max_step_launcher(float* d_values,
                                  int64_t* d_indices,
                                  const float* d_input,
                                  int N, int H, int W,
                                  cudaStream_t stream) {
    size_t total_out = (size_t)N * H * W * 2;
    int threads = 256;
    int blocks = (int)((total_out + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    max_step_kernel<<<blocks, threads, 0, stream>>>(d_input, d_values, d_indices, N, H, W);
}

extern "C" void view_as_complex_launcher(cuComplex* d_out,
                                         const float* d_in,
                                         int N, int H, int W,
                                         cudaStream_t stream) {
    size_t total_out = (size_t)N * H * W;
    int threads = 256;
    int blocks = (int)((total_out + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    view_as_complex_kernel<<<blocks, threads, 0, stream>>>(d_in, d_out, N, H, W);
}
