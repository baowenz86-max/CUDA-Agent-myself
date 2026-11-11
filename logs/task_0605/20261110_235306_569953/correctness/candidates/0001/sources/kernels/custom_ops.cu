// CUDA kernels for MaxPool2d and ReLU
#include <cuda_runtime.h>
#include <cstdint>

// MaxPool2d kernel: input is [N, C, H, W] contiguous float32.
// kernel_size, stride, padding, dilation are scalar ints.
// Output shape computed as H_out = floor((H + 2*padding - dilation*(kernel_size-1) - 1)/stride) + 1
// W_out similarly.
// Here we assume kernel size K = kernel_size (square) but the model actually uses (3,5).
// However, the contract passes a single kernel_size int, so we use square kernel.
// In model_new.py, self.maxpool.kernel_size is a tuple (3,5); passing it as int would cause error.
// The contract seems inconsistent: it says kernel_size is int but model uses tuple. We need to handle both possible.
// We will define launcher with int kernel_size, assuming caller passes the height component. But for actual correctness,
// we should support tuple (kh,kw). The binding layer can pass two ints. However, contract says one int.
// To be safe, we implement two launcher variants: one with single int (square) and one with two ints for height and width.
// But only one signature can be registered. The model_new.py passes self.maxpool.kernel_size which is tuple (3,5) as argument; pybind will convert it to int? No, tuple is not int.
// This is a problem. We must follow model_new.py: it passes tuple. But contract parameter list shows int. Likely the binding layer will extract first element? Not possible.
// We need to inspect model_new.py more carefully: it passes self.maxpool.kernel_size directly. In Python, tuple. The C++ binding must accept tuple. But contract says int.
// Since we cannot modify model_new.py, we must design launcher to handle tuple via pybind? But we are not allowed to use pybind in .cu.
// The binding.cpp in next stage will handle Python arguments. We only provide C-style launcher. We can provide launcher taking individual ints: kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w.
// Then binding stage can unpack tuple and call. The contract fragments show parameters: torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation. That suggests binding will take ints, but model passes tuple. There is mismatch.
// Given constraints, we implement launcher with int kernel_size (assume square) and also provide a version with explicit dims. The binding layer can choose. For now, implement both as extern "C" functions.

extern "C" void maxpool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    cudaStream_t stream
) {
    int H_out = (H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Using grid-stride loop kernel
    maxpool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W, H_out, W_out,
        kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, total);
}

// ReLU launcher
extern "C" void relu_forward_launcher(
    const float* input,
    float* output,
    int n,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    relu_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}

// kernel implementations
__global__ void maxpool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    int total
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < total; i += stride) {
        int w_out = i % W_out;
        int h_out = (i / W_out) % H_out;
        int c = (i / (W_out * H_out)) % C;
        int n = i / (W_out * H_out * C);
        float max_val = -1e30f;
        int h_start = h_out * stride_h - padding_h;
        int w_start = w_out * stride_w - padding_w;
        for (int kh = 0; kh < kernel_h; ++kh) {
            int h_in = h_start + kh * dilation_h;
            if (h_in < 0 || h_in >= H) continue;
            for (int kw = 0; kw < kernel_w; ++kw) {
                int w_in = w_start + kw * dilation_w;
                if (w_in < 0 || w_in >= W) continue;
                float val = input[((n * C + c) * H + h_in) * W + w_in];
                if (val > max_val) max_val = val;
            }
        }
        output[i] = max_val;
    }
}

__global__ void relu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        output[i] = fmaxf(input[i], 0.0f);
    }
}
