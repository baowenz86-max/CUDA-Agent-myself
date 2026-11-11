#include <cuda_runtime.h>

// Hardshrink kernel: out = x if x > lambd else x if x < -lambd else 0
__global__ void hardshrink_kernel(const float* __restrict__ x, float* __restrict__ out, float lambd, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float val = x[idx];
        out[idx] = (val > lambd || val < -lambd) ? val : 0.0f;
    }
}

// MaxPool2d kernel: one thread per output element
// Supports arbitrary strides via input tensor shape and stride parameters passed as integers (contiguous decomposition)
// Input assumed to be contiguous in memory; if not, binding should ensure contiguous copy.
// However, launcher receives raw pointer and strides as int64_t for generality,
// but for simplicity we assume contiguous in the kernel below; we'll adapt launcher to handle non-contiguous via strides.
// Actually, to support non-contiguous input as per contract, we can pass full stride array or flatten in binding.
// Binding will ensure contiguous by calling .contiguous() if needed, so kernel can assume contiguous.
// Thus kernel uses linear index mapping with regular strides from shape.
__global__ void max_pool2d_kernel(const float* __restrict__ input, float* __restrict__ output,
                                  int N, int C, int H_in, int W_in,
                                  int kernel_h, int kernel_w, int stride_h, int stride_w,
                                  int pad_h, int pad_w, int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * C * H_out * W_out;
    if (idx >= total_outputs) return;

    // Decode linear index to (n,c,h_out,w_out)
    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;

    int h_start = h_out * stride_h - pad_h;
    int w_start = w_out * stride_w - pad_w;
    float max_val = -3.402823466e+38f; // -FLT_MAX

    for (int kh = 0; kh < kernel_h; ++kh) {
        int h_in = h_start + kh;
        if (h_in < 0 || h_in >= H_in) continue;
        for (int kw = 0; kw < kernel_w; ++kw) {
            int w_in = w_start + kw;
            if (w_in < 0 || w_in >= W_in) continue;
            int input_idx = ((n * C + c) * H_in + h_in) * W_in + w_in;
            float val = input[input_idx];
            if (val > max_val) max_val = val;
        }
    }
    output[idx] = max_val;
}

// Launcher for hard shrink
extern "C" void hardshrink_launcher(float* x, float lambd, int total_elements, cudaStream_t stream) {
    const int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;
    hardshrink_kernel<<<grid_size, block_size, 0, stream>>>(x, x, lambd, total_elements);
}

// Launcher for max pool 2D
extern "C" void max_pool2d_launcher(const float* x, float* out,
                                    int N, int C, int H_in, int W_in,
                                    int kernel_h, int kernel_w, int stride_h, int stride_w,
                                    int pad_h, int pad_w, int H_out, int W_out,
                                    cudaStream_t stream) {
    int total_outputs = N * C * H_out * W_out;
    const int block_size = 256;
    int grid_size = (total_outputs + block_size - 1) / block_size;
    max_pool2d_kernel<<<grid_size, block_size, 0, stream>>>(
        x, out, N, C, H_in, W_in, kernel_h, kernel_w,
        stride_h, stride_w, pad_h, pad_w, H_out, W_out);
}
