// CUDA kernels for reflection_pad2d, ge_scalar, remainder_scalar
#include <cuda_runtime.h>

// Helper to reflect index in [0, dim-1]
__device__ __forceinline__ int reflect_index(int idx, int dim) {
    if (idx < 0) return -idx;
    if (idx >= dim) return 2 * dim - idx - 2;
    return idx;
}

// Kernel for reflection padding 2D with arbitrary padding sizes and strides
__global__ void reflection_pad2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int in_stride_n, int in_stride_c, int in_stride_h, int in_stride_w,
    int out_stride_n, int out_stride_c, int out_stride_h, int out_stride_w,
    int out_height, int out_width) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * out_height * out_width;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total; i += stride) {
        // Decode output index
        int w = i % out_width;
        int tmp = i / out_width;
        int h = tmp % out_height;
        tmp /= out_height;
        int c = tmp % C;
        int n = tmp / C;

        // Compute corresponding input coordinates with reflection
        int in_h = h - pad_top;
        int in_w = w - pad_left;
        in_h = reflect_index(in_h, H);
        in_w = reflect_index(in_w, W);

        // Read input value using strides
        const float* in_ptr = input + n * in_stride_n + c * in_stride_c + in_h * in_stride_h + in_w * in_stride_w;
        float val = *in_ptr;

        // Write output
        float* out_ptr = output + n * out_stride_n + c * out_stride_c + h * out_stride_h + w * out_stride_w;
        *out_ptr = val;
    }
}

// Kernel for ge_scalar: input >= scalar, output bool (as unsigned char)
__global__ void ge_scalar_kernel(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    float scalar,
    int total_elements,
    int in_stride_n, int in_stride_c, int in_stride_h, int in_stride_w) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        // Decode index assuming contiguous output? But input may be non-contiguous.
        // However, ge_scalar is applied on the padded tensor which is contiguous from our launcher.
        // For generality, we handle non-contiguous by using strides.
        // We assume output is contiguous with same shape. We need shape info: but total_elements suffices for flat index.
        int w = i % W;
        int tmp = i / W;
        int h = tmp % H;
        tmp /= H;
        int c = tmp % C;
        int n = tmp / C;

        // Here we need shapes and strides. We'll pass them.
        const float* in_ptr = input + n * in_stride_n + c * in_stride_c + h * in_stride_h + w * in_stride_w;
        float val = *in_ptr;
        unsigned char result = (val >= scalar) ? 1 : 0;
        output[i] = result;
    }
}

// Kernel for remainder_scalar: input (bool or int) % scalar, output int64.
// We operate on bool input (as unsigned char) and output int64 (long long).
__global__ void remainder_scalar_kernel(
    const unsigned char* __restrict__ input,
    long long* __restrict__ output,
    double scalar,
    int total_elements) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < total_elements; i += stride) {
        // input is bool (0 or 1), scalar is 2.0 -> result 0 or 1 (as int64)
        unsigned char val = input[i];
        // Use Python-like modulo: result = val - floor(val/scalar)*scalar
        // For val in {0,1}, scalar=2, result is 0 or 1.
        double dval = (double)val;
        double rem = dval - floor(dval / scalar) * scalar;
        output[i] = (long long)rem;
    }
}

// Launchers (extern "C")

extern "C" void reflection_pad2d_forward(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    int out_height, int out_width,
    cudaStream_t stream) {

    int total = N * C * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Cap blocks to avoid excessive grid launch; use at most 65535 blocks for older GPUs
    if (blocks > 65535) blocks = 65535;
    reflection_pad2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W,
        pad_left, pad_right, pad_top, pad_bottom,
        (int)in_stride_n, (int)in_stride_c, (int)in_stride_h, (int)in_stride_w,
        (int)out_stride_n, (int)out_stride_c, (int)out_stride_h, (int)out_stride_w,
        out_height, out_width);
}

extern "C" void ge_scalar_forward(
    const float* input,
    unsigned char* output,
    float scalar,
    int total_elements,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    int H, int W, int C,
    cudaStream_t stream) {

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    ge_scalar_kernel<<<blocks, threads, 0, stream>>>(
        input, output, scalar, total_elements,
        (int)in_stride_n, (int)in_stride_c, (int)in_stride_h, (int)in_stride_w,
        H, W, C);
}

extern "C" void remainder_scalar_forward(
    const unsigned char* input,
    long long* output,
    double scalar,
    int total_elements,
    cudaStream_t stream) {

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    remainder_scalar_kernel<<<blocks, threads, 0, stream>>>(
        input, output, scalar, total_elements);
}
