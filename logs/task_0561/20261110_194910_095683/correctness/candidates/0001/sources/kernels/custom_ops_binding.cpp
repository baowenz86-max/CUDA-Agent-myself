#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu (C linkage)
extern "C" void custom_nan_to_zero_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream);

extern "C" void custom_prelu_launcher(
    float* output,
    const float* input,
    const float* weight,
    int64_t numel,
    int64_t channels,
    int64_t hw,
    cudaStream_t stream);

torch::Tensor custom_nan_to_zero(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_nan_to_zero_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        stream);

    return output;
}

torch::Tensor custom_prelu(torch::Tensor x, torch::Tensor weight) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1), "Weight must be 1-D with size equal to x's channel dimension");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    int64_t channels = x.size(1);
    int64_t hw = x.size(2) * x.size(3); // height * width

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_prelu_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        numel,
        channels,
        hw,
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_nan_to_zero", &custom_nan_to_zero, "Replace NaN with zero", py::arg("x"));
    m.def("custom_prelu", &custom_prelu, "PReLU activation with per-channel weight", py::arg("x"), py::arg("weight"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
