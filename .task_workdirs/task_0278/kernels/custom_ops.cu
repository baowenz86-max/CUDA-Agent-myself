#include <cuda_runtime.h>
#include <math.h>

// ELU forward kernel: y = x > 0 ? x : alpha * (exp(x) - 1)
__global__ void elu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long total_elements,
    float alpha) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = idx; i < total_elements; i += stride) {
        float x = input[i];
        output[i] = (x > 0.0f) ? x : alpha * (expf(x) - 1.0f);
    }
}

// MaxPool2d forward kernel for kernel_size=(2,2), stride=(2,2), padding=0, dilation=1, ceil_mode=false
// Input: 4D tensor (N, C, H, W), float32
// Output: 4D tensor (N, C, H_out, W_out)
// We assume input is contiguous; if not, launcher will make a contiguous copy.
__global__ void max_pool2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out) {
    int total_outputs = N * C * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_outputs) return;

    // Convert linear index to (n, c, h_out, w_out)
    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;

    int h_start = h_out * 2; // stride = 2, padding = 0
    int w_start = w_out * 2;

    const float* input_nc = input + ((long long)n * C + c) * H * W;
    float max_val = -INFINITY;
    for (int dh = 0; dh < 2; ++dh) {
        int h_idx = h_start + dh;
        if (h_idx >= H) continue;
        for (int dw = 0; dw < 2; ++dw) {
            int w_idx = w_start + dw;
            if (w_idx >= W) continue;
            float val = input_nc[(long long)h_idx * W + w_idx];
            if (val > max_val) max_val = val;
        }
    }
    output[idx] = max_val;
}

// Helper to check if a float tensor is contiguous (all strides are standard)
// But since we don't have Tensor object, we pass a flag from binding.
// In this .cu we only implement launchers that assume contiguous input,
// and binding should pass a contiguous copy if needed.
extern "C" void custom_elu_forward_launcher(
    const float* input,
    float* output,
    long long total_elements,
    float alpha,
    cudaStream_t stream) {
    const int block_size = 256;
    long long blocks = (total_elements + block_size - 1) / block_size;
    if (blocks > 2147483647) {
        // For very large tensors, use grid-stride loop with max grid size
        int max_grid = 2147483647 / block_size;
        if (max_grid < 1) max_grid = 1;
        blocks = max_grid;
        elu_forward_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
            input, output, total_elements, alpha);
    } else {
        elu_forward_kernel<<<(unsigned int)blocks, block_size, 0, stream>>>(
            input, output, total_elements, alpha);
    }
}

extern "C" void custom_max_pool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    cudaStream_t stream) {
    int total_outputs = N * C * H_out * W_out;
    const int block_size = 256;
    int blocks = (total_outputs + block_size - 1) / block_size;
    max_pool2d_forward_kernel<<<blocks, block_size, 0, stream>>>(
        input, output, N, C, H, W, H_out, W_out);
}
