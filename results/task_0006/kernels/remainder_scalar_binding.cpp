#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

namespace py = pybind11;

extern "C" void remainder_scalar_launcher(
    const bool* input,
    int64_t* output,
    int64_t divisor,
    long total_elements,
    cudaStream_t stream
);

torch::Tensor remainder_scalar_forward(torch::Tensor input, int64_t divisor) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kBool, "Only bool input is implemented");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kInt64));

    long total_elements = input.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    remainder_scalar_launcher(
        input.data_ptr<bool>(),
        output.data_ptr<int64_t>(),
        divisor,
        total_elements,
        stream
    );

    return output;
}

void register_remainder_scalar(py::module& m) {
    m.def("remainder_scalar_forward", &remainder_scalar_forward,
          "Remainder scalar forward",
          py::arg("input"), py::arg("divisor"));
}

REGISTER_BINDING(remainder_scalar, register_remainder_scalar);
