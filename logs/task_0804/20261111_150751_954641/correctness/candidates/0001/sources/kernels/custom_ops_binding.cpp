#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void stack_forward_launcher(
    const float* x1,
    const float* x2,
    float* out,
    int64_t numel_per_input,
    cudaStream_t stream);

torch::Tensor stack_forward(torch::Tensor x1, torch::Tensor x2) {
    // Validate inputs
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(x1.is_contiguous(), "x1 must be contiguous");
    TORCH_CHECK(x2.is_contiguous(), "x2 must be contiguous");
    TORCH_CHECK(x1.sizes() == x2.sizes(), "x1 and x2 must have the same shape");

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Allocate output tensor of shape [2, ...x1.shape]
    std::vector<int64_t> out_shape(1, 2);
    out_shape.insert(out_shape.end(), x1.sizes().begin(), x1.sizes().end());
    auto out = torch::empty(out_shape, x1.options());

    TORCH_CHECK(out.is_contiguous(), "Output must be contiguous");

    int64_t numel_per_input = x1.numel();

    // Call launcher
    stack_forward_launcher(
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        out.data_ptr<float>(),
        numel_per_input,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("stack_forward", &stack_forward,
          "Stack two CUDA float32 tensors along a new first dimension",
          py::arg("x1"), py::arg("x2"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
