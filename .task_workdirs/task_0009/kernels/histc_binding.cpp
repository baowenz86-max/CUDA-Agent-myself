#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from .cu file
extern "C" void histc_forward_launcher(
    const float* input,
    long long num_elements,
    int bins,
    float min_val,
    float max_val,
    float* output,
    cudaStream_t stream
);

namespace py = pybind11;

// Binding wrapper
torch::Tensor histc_forward(torch::Tensor input, int64_t bins, double min_val, double max_val) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(bins > 0, "bins must be positive");

    auto output = torch::zeros({bins}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    histc_forward_launcher(
        input.data_ptr<float>(),
        input.numel(),
        (int)bins,
        (float)min_val,
        (float)max_val,
        output.data_ptr<float>(),
        stream
    );

    return output;
}

void register_histc(pybind11::module& m) {
    m.def("histc_forward", &histc_forward,
          "Histogram forward",
          py::arg("input"), py::arg("bins"), py::arg("min"), py::arg("max"));
}

REGISTER_BINDING(histc, register_histc);
