#include <cuda_runtime.h>
#include <stdint.h>

// This file contains CUDA kernels and host launchers for custom ops used by the model.
// It must not depend on torch, ATen, or pybind. All tensor handling is done in the binding layer.

// We use cudaMemcpyAsync for simple contiguous copies to avoid any index errors.

// Launcher for unbind_dim1. Since unbind returns views, no CUDA work is needed.
// The binding layer will create Python tuple of tensor views using torch::select.
// We provide a no-op launcher for completeness.
extern "C" void unbind_dim1_launcher(const float* x,
                                     int64_t B,
                                     int64_t C,
                                     int64_t H,
                                     int64_t W,
                                     cudaStream_t stream) {
    (void)x; (void)B; (void)C; (void)H; (void)W; (void)stream;
    // No kernel launch; views are handled in C++.
}

// Launcher for identity_new_tensor: copies data to a new contiguous tensor.
extern "C" void identity_copy_launcher(const float* input,
                                        float* output,
                                        int64_t numel,
                                        cudaStream_t stream) {
    if (numel <= 0) return;
    // Use cudaMemcpyAsync for reliable copy
    cudaMemcpyAsync(output, input, numel * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

// Launcher for stack_dim1: stacks C contiguous tensors into output [B,C,H,W].
// We launch one copy per input tensor using cudaMemcpyAsync.
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
    for (int64_t c = 0; c < C; ++c) {
        cudaMemcpyAsync(output + c * per_tensor,
                        input_ptrs[c],
                        per_tensor * sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        stream);
    }
}
