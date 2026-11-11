#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void gelu_forward_launcher(
    float* output,
    const float* input,
    int64_t total_elements,
    int ndim,
    const int64_t* sizes,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream);

extern "C" void selu_forward_launcher(
    float* output,
    const float* input,
    int64_t total_elements,
    int ndim,
    const int64_t* sizes,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream);

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int ndim = x.dim();
    auto sizes = x.sizes();
    auto input_strides = x.strides();
    auto output_strides = output.strides();

    gelu_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        ndim,
        sizes.data(),
        input_strides.data(),
        output_strides.data(),
        stream);

    return output;
}

torch::Tensor selu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int ndim = x.dim();
    auto sizes = x.sizes();
    auto input_strides = x.strides();
    auto output_strides = output.strides();

    selu_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        ndim,
        sizes.data(),
        input_strides.data(),
        output_strides.data(),
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("gelu_forward", &gelu_forward, "GELU forward (exact erf)",
          py::arg("x"));
    m.def("selu_forward", &selu_forward, "SELU forward",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
