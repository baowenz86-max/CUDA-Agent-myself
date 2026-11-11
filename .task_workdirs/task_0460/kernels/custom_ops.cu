#include <cuda_runtime.h>
#include <stdint.h>

// This file contains CUDA kernels and host launchers for custom ops used by the model.
// It must not depend on torch, ATen, or pybind. All tensor handling is done in the binding layer.

// Kernel: copies input contiguous buffer to output contiguous buffer.
// Used by identity_new_tensor: copy [512,128,128] contiguous tensor.
// grid-stride loop for generality.
__global__ void copy_kernel_float(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < numel; i += stride) {
        output[i] = input[i];
    }
}

// Kernel: stacks a list of contiguous tensors along a new dimension 1.
// Input: each tensor has shape [B, H, W] -> flattened per tensor: B*H*W
// Output: shape [B, C, H, W], where C is number of tensors.
// We assume all input tensors are contiguous and have same shape (B,H,W).
// Output is contiguous.
__global__ void stack_kernel_float(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int64_t B,
                                   int64_t C,
                                   int64_t H,
                                   int64_t W) {
    int64_t total = B * C * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t per_tensor = B * H * W;
    for (int64_t i = idx; i < total; i += stride) {
        // decompose output index: i = (((b * C + c) * H + h) * W + w)
        int64_t tmp = i;
        int64_t w = tmp % W;
        tmp /= W;
        int64_t h = tmp % H;
        tmp /= H;
        int64_t c = tmp % C;
        int64_t b = tmp / C;
        int64_t src_idx = (b * H + h) * W + w;
        output[i] = input[c * per_tensor + src_idx];
    }
}

// Launcher for unbind_dim1. Since unbind returns views, no CUDA work is needed.
// The binding layer will create Python tuple of tensor views using torch::unbind-like slicing.
// We provide a no-op launcher for completeness; it takes raw pointer and does nothing.
extern "C" void unbind_dim1_launcher(const float* x,
                                     int64_t B,
                                     int64_t C,
                                     int64_t H,
                                     int64_t W,
                                     cudaStream_t stream) {
    // No kernel launch; views are handled in C++.
    (void)x; (void)B; (void)C; (void)H; (void)W; (void)stream;
}

// Launcher for identity_new_tensor: copies data to a new contiguous tensor.
extern "C" void identity_copy_launcher(const float* input,
                                        float* output,
                                        int64_t numel,
                                        cudaStream_t stream) {
    if (numel <= 0) return;
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    // Limit blocks to a reasonable max to avoid excessive grid dimensions.
    if (blocks > 65535) blocks = 65535;
    copy_kernel_float<<<blocks, threads, 0, stream>>>(input, output, numel);
}

// Launcher for stack_dim1: stacks C contiguous tensors into output [B,C,H,W].
// To keep the launcher simple and avoid device pointer array allocation,
// we launch one copy kernel per input tensor, writing its data to the correct output slice.
// The binding layer passes a host pointer to an array of device pointers.
extern "C" void stack_dim1_launcher(const float* const* input_ptrs,
                                    int64_t B,
                                    int64_t C,
                                    int64_t H,
                                    int64_t W,
                                    float* output,
                                    cudaStream_t stream) {
    if (B <= 0 || C <= 0 || H <= 0 || W <= 0) return;
    int64_t per_tensor = B * H * W;
    int threads = 256;
    int blocks = (int)((per_tensor + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    for (int64_t c = 0; c < C; ++c) {
        copy_kernel_float<<<blocks, threads, 0, stream>>>(
            input_ptrs[c], output + c * per_tensor, per_tensor);
    }
}

// Note: The unbind_dim1_launcher is a no-op because unbind returns views, handled in binding.
// The binding layer will create a tuple of tensors by slicing the original tensor using as_strided or narrow.
// No CUDA kernel is required for unbind.
