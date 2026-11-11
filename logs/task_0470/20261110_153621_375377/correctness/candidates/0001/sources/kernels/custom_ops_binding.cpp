// Binding file for custom_abs operator
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void custom_abs_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream);

// PyTorch wrapper function
torch::Tensor custom_abs_forward(torch::Tensor x) {
    // Check CUDA tensor
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    // Check dtype
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    // Ensure contiguous (the kernel assumes contiguous layout)
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    // Allocate output tensor with same options
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call the launcher
    custom_abs_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_abs(pybind11::module& m) {
    m.def("custom_abs", &custom_abs_forward, "Element-wise absolute value (CUDA)",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_abs, register_custom_abs);
