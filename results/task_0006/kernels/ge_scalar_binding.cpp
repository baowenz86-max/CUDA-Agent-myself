#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

namespace py = pybind11;

extern "C" void ge_scalar_launcher(
    const float* input,
    bool* output,
    float scalar,
    long total_elements,
    cudaStream_t stream
);

torch::Tensor ge_scalar_forward(torch::Tensor input, double scalar) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Only float32 is implemented");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kBool));

    long total_elements = input.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    ge_scalar_launcher(
        input.data_ptr<float>(),
        output.data_ptr<bool>(),
        (float)scalar,
        total_elements,
        stream
    );

    return output;
}

void register_ge_scalar(py::module& m) {
    m.def("ge_scalar_forward", &ge_scalar_forward,
          "Greater than or equal scalar forward",
          py::arg("input"), py::arg("scalar"));
}

REGISTER_BINDING(ge_scalar, register_ge_scalar);
