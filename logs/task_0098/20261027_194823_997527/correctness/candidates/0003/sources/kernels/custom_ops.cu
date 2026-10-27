// CUDA kernels for reflection_pad2d and elementwise max/min
// All launchers use raw pointers, scalar and cudaStream_t only.
// Binding wrappers are implemented separately in *_binding.cpp.

#include <cuda_runtime.h>
#include <cstdint>

// Reflection index helper: reflect x according to ReflectionPad rules
__device__ __forceinline__ int reflect_index(int x, int size) {
    if (x < 0) {
        return -x;
    } else if (x >= size) {
        return 2 * (size - 1) - x;
    } else {
        return x;
    }
}

// Kernel for ReflectionPad2d. Input is a 4D tensor with arbitrary strides.
// Output is contiguous with shape [N, C, H_out, W_out].
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out,
    int pad,
    int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w) {

    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t total = N * C * H_out * W_out;

    // Grid-stride loop to cover all elements even if grid is capped
    for (; idx < total; idx += stride) {
        int64_t w_out = idx % W_out;
        int64_t h_out = (idx / W_out) % H_out;
        int64_t c = (idx / (W_out * H_out)) % C;
        int64_t n = idx / (W_out * H_out * C);

        int h_in = reflect_index(h_out - pad, H);
        int w_in = reflect_index(w_out - pad, W);

        int64_t input_idx = n * stride_n + c * stride_c + h_in * stride_h + w_in * stride_w;
        output[idx] = input[input_idx];
    }
}

// Generic elementwise max kernel for float32 tensors, assumes same shape and contiguous layout.
__global__ void elementwise_max_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t numel) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; i < numel; i += stride) {
        out[i] = fmaxf(a[i], b[i]);
    }
}

// Generic elementwise min kernel for float32 tensors, assumes same shape and contiguous layout.
__global__ void elementwise_min_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t numel) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; i < numel; i += stride) {
        out[i] = fminf(a[i], b[i]);
    }
}

// Launcher for ReflectionPad2d (4D input, symmetric pad)
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
    int pad,
    cudaStream_t stream) {

    int64_t H_out = H + 2 * pad;
    int64_t W_out = W + 2 * pad;
    int64_t total = N * C * H_out * W_out;
    int threads = 256;
    int max_blocks = 65535;
    int blocks = (int)min((int64_t)max_blocks, (total + threads - 1) / threads);
    if (blocks <= 0) blocks = 1;
    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, H_out, W_out, pad,
        stride_n, stride_c, stride_h, stride_w);
}

// Launcher for elementwise max (same shape, contiguous)
extern "C" void elementwise_max_launcher(
    const float* a,
    const float* b,
    float* out,
    int64_t numel,
    cudaStream_t stream) {

    int threads = 256;
    int blocks = (int)min((int64_t)65535, (numel + threads - 1) / threads);
    if (blocks <= 0) blocks = 1;
    elementwise_max_kernel<<<blocks, threads, 0, stream>>>(a, b, out, numel);
}

// Launcher for elementwise min (same shape, contiguous)
extern "C" void elementwise_min_launcher(
    const float* a,
    const float* b,
    float* out,
    int64_t numel,
    cudaStream_t stream) {

    int threads = 256;
    int blocks = (int)min((int64_t)65535, (numel + threads - 1) / threads);
    if (blocks <= 0) blocks = 1;
    elementwise_min_kernel<<<blocks, threads, 0, stream>>>(a, b, out, numel);
}
