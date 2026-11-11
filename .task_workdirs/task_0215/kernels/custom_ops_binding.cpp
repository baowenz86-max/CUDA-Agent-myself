// PyTorch binding for custom tanh CUDA kernel
// This file handles tensor validation, memory allocation, stream management,
// and Python registration. The actual CUDA kernel is in custom_ops.cu.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declared in custom_ops.cu
extern "C" void custom_tanh_launcher(float* output,
                                     const float* input,
                                     int64_t numel,
                                     cudaStream_t stream);

// PyTorch wrapper for the custom tanh operation.
// Input: x - CUDA tensor of dtype float32, any shape acceptable but must be finite.
// Output: new tensor of same shape, dtype, device, computed tanh elementwise.
torch::Tensor custom_tanh(torch::Tensor x) {
    // Validate device
    TORCH_CHECK(x.is_cuda(), "custom_tanh: input must be a CUDA tensor");
    // Validate dtype
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "custom_tanh: input must be float32");
    // Ensure contiguous layout for linear element access in kernel
    auto x_contig = x.contiguous();

    // Allocate output with same shape as contiguous input (identical to original shape)
    auto output = torch::empty_like(x_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    custom_tanh_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        x_contig.numel(),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_tanh: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function called by BindingRegistry
void register_custom_tanh(pybind11::module& m) {
    m.def("custom_tanh", &custom_tanh,
          "Custom Tanh (CUDA)",
          pybind11::arg("x"));
}

// Auto-register binding
REGISTER_BINDING(custom_tanh, register_custom_tanh);
