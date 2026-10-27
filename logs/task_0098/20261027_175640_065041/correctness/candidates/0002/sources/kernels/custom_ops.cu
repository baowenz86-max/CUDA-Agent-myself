// CUDA kernels for reflection_pad2d, elementwise_max, elementwise_minimum
#include <cuda_runtime.h>

// Reflection padding indexing:
// pad = pad_size
// input index: i_in = i_out - pad
// if i_in < 0: i_in = -i_in
// if i_in >= in_size: i_in = 2 * (in_size - 1) - i_in
__device__ __forceinline__ int reflect_index(int idx, int pad, int in_size) {
    int i = idx - pad;
    if (i < 0) {
        i = -i;
    } else if (i >= in_size) {
        i = 2 * (in_size - 1) - i;
    }
    return i;
}

// ReflectionPad2d kernel
// input: (N, C, H_in, W_in) with strides sn, sc, sh, sw
// output: (N, C, H_out, W_out) with H_out = H_in + 2*pad, W_out = W_in + 2*pad
// output is contiguous, with row-major strides: C*H_out*W_out, H_out*W_out, W_out, 1
template <typename T>
__global__ void reflection_pad2d_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out, int pad,
    long long sn, long long sc, long long sh, long long sw) {

    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (out_idx >= total) return;

    // Decompose linear index
    int w_out = out_idx % W_out;
    int tmp = out_idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;

    int h_in = reflect_index(h_out, pad, H_in);
    int w_in = reflect_index(w_out, pad, W_in);

    // Compute input linear index using strides
    long long in_linear = (long long)n * sn + (long long)c * sc + (long long)h_in * sh + (long long)w_in * sw;

    output[out_idx] = input[in_linear];
}

// Elementwise max kernel for contiguous tensors (same shape)
template <typename T>
__global__ void elementwise_max_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out,
    int total) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = (a[idx] > b[idx]) ? a[idx] : b[idx];
}

// Elementwise min kernel for contiguous tensors (same shape)
template <typename T>
__global__ void elementwise_min_kernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out,
    int total) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = (a[idx] < b[idx]) ? a[idx] : b[idx];
}

// Launcher for reflection_pad2d
extern "C" void reflection_pad2d_launcher(
    const float* input, float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out, int pad,
    long long sn, long long sc, long long sh, long long sw,
    cudaStream_t stream) {

    int total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reflection_pad2d_kernel<float><<<blocks, threads, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out, pad,
        sn, sc, sh, sw);
}

// Launcher for elementwise_max (assumes contiguous a, b, out)
extern "C" void elementwise_max_launcher(
    const float* a, const float* b, float* out,
    int total,
    cudaStream_t stream) {

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    elementwise_max_kernel<float><<<blocks, threads, 0, stream>>>(a, b, out, total);
}

// Launcher for elementwise_minimum (assumes contiguous a, b, out)
extern "C" void elementwise_minimum_launcher(
    const float* a, const float* b, float* out,
    int total,
    cudaStream_t stream) {

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    elementwise_min_kernel<float><<<blocks, threads, 0, stream>>>(a, b, out, total);
}
