#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher functions from custom_ops.cu
extern "C" void abs_forward_launcher(const float* input, float* output, long long numel, cudaStream_t stream);
extern "C" void pow_scalar_int_launcher(const float* input, float* output, long long numel, int exponent, cudaStream_t stream);

// Wrapper for abs_forward
torch::Tensor abs_forward(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Allocate output tensor
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    abs_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );

    return output;
}

// Wrapper for pow_scalar_forward
torch::Tensor pow_scalar_forward(torch::Tensor x, int64_t exponent) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Allocate output tensor
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher (exponent is int, contract expects int64_t but launcher uses int)
    pow_scalar_int_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        static_cast<int>(exponent),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward, "Elementwise absolute value",
          py::arg("x"));
    m.def("pow_scalar_forward", &pow_scalar_forward, "Elementwise power with scalar exponent",
          py::arg("x"), py::arg("exponent") = 3);
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
