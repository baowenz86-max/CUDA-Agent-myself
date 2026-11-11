#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void hardtanh_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    float min_val,
    float max_val,
    cudaStream_t stream);

extern "C" void div_scalar_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    float divisor,
    cudaStream_t stream);

// PyTorch wrapper for hardtanh
torch::Tensor hardtanh_forward(torch::Tensor input, double min_val = -1.0, double max_val = 1.0) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    // Ensure contiguous (launcher assumes contiguous)
    auto input_contig = input.contiguous();

    // Allocate output with same shape and options
    auto output = torch::empty_like(input_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call CUDA launcher
    hardtanh_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<long long>(input_contig.numel()),
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        stream
    );

    return output;
}

// PyTorch wrapper for div_scalar
torch::Tensor div_scalar_forward(torch::Tensor input, double divisor) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    // Ensure contiguous
    auto input_contig = input.contiguous();

    // Allocate output
    auto output = torch::empty_like(input_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call CUDA launcher
    div_scalar_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<long long>(input_contig.numel()),
        static_cast<float>(divisor),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("hardtanh_forward", &hardtanh_forward,
          "Apply hardtanh to input tensor",
          py::arg("input"),
          py::arg("min_val") = -1.0,
          py::arg("max_val") = 1.0);

    m.def("div_scalar_forward", &div_scalar_forward,
          "Divide input tensor by scalar",
          py::arg("input"),
          py::arg("divisor"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
