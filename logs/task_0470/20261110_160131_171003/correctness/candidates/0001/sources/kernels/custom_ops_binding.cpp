#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cstring>
#include <cmath>
#include "../binding_registry.h"

// Since this model operates entirely on CPU tensors, no CUDA kernels are required.
// The bindings implement the required operations directly on CPU memory.

// Copies input tensor to target device. Here target is always CPU, so just return a contiguous copy.
torch::Tensor to_device_forward(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cpu(), "Input must be a CPU tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    // Return a copy of the tensor, preserving shape and dtype.
    auto output = torch::empty_like(x);
    std::memcpy(output.data_ptr<float>(), x.data_ptr<float>(), x.numel() * sizeof(float));
    return output;
}

// Computes element-wise absolute value.
torch::Tensor abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cpu(), "Input must be a CPU tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    auto output = torch::empty_like(x);
    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    int64_t n = x.numel();
    for (int64_t i = 0; i < n; ++i) {
        output_ptr[i] = std::abs(input_ptr[i]);
    }
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("to_device_forward", &to_device_forward, "Copy tensor to CPU (identity for CPU input)", py::arg("x"));
    m.def("abs_forward", &abs_forward, "Element-wise absolute value", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
