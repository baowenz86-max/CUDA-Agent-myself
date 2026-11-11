#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void abs_forward_launcher(
    float* output,
    const float* input,
    const long long* shape,
    const long long* input_strides,
    const long long* output_strides,
    int ndim,
    long long numel,
    cudaStream_t stream
);

extern "C" void hardsigmoid_forward_launcher(
    float* output,
    const float* input,
    const long long* shape,
    const long long* input_strides,
    const long long* output_strides,
    int ndim,
    long long numel,
    cudaStream_t stream
);

namespace {

torch::Tensor abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");

    auto output = torch::empty_like(x);

    int ndim = x.dim();
    TORCH_CHECK(ndim <= 8, "Only up to 8 dimensions are supported");

    std::vector<long long> shape(ndim);
    std::vector<long long> input_strides(ndim);
    std::vector<long long> output_strides(ndim);
    for (int i = 0; i < ndim; ++i) {
        shape[i] = x.size(i);
        input_strides[i] = x.stride(i);
        output_strides[i] = output.stride(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    abs_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        shape.data(),
        input_strides.data(),
        output_strides.data(),
        ndim,
        x.numel(),
        stream
    );

    return output;
}

torch::Tensor hardsigmoid_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");

    auto output = torch::empty_like(x);

    int ndim = x.dim();
    TORCH_CHECK(ndim <= 8, "Only up to 8 dimensions are supported");

    std::vector<long long> shape(ndim);
    std::vector<long long> input_strides(ndim);
    std::vector<long long> output_strides(ndim);
    for (int i = 0; i < ndim; ++i) {
        shape[i] = x.size(i);
        input_strides[i] = x.stride(i);
        output_strides[i] = output.stride(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    hardsigmoid_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        shape.data(),
        input_strides.data(),
        output_strides.data(),
        ndim,
        x.numel(),
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward, "Absolute value forward",
          py::arg("x"));
    m.def("hardsigmoid_forward", &hardsigmoid_forward, "Hardsigmoid forward",
          py::arg("x"));
}

} // namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
