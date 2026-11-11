#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_abs_forward_launcher(
    const float* input,
    float* output,
    long long n,
    cudaStream_t stream
);

extern "C" void custom_clamp_forward_launcher(
    const float* input,
    float* output,
    double min_value,
    double max_value,
    long long n,
    cudaStream_t stream
);

// Wrapper for custom_abs_forward
torch::Tensor custom_abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_abs_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<long long>(x.numel()),
        stream
    );

    return output;
}

// Wrapper for custom_clamp_forward
torch::Tensor custom_clamp_forward(torch::Tensor x, double min_value, double max_value) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_clamp_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        min_value,
        max_value,
        static_cast<long long>(x.numel()),
        stream
    );

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_abs_forward", &custom_abs_forward, "Element-wise absolute value",
          py::arg("x"));
    m.def("custom_clamp_forward", &custom_clamp_forward, "Clamp tensor to [min_value, max_value]",
          py::arg("x"), py::arg("min_value"), py::arg("max_value"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
