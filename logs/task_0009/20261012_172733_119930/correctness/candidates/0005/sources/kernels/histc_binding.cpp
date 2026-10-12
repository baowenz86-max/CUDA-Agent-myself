// PyTorch binding for histc custom op
#include <torch/types.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declaration of launcher from histc.cu
extern "C" void histc_forward_launcher(
    float* hist,
    const float* input,
    int64_t numel,
    int bins,
    float min,
    float max,
    cudaStream_t stream);

torch::Tensor histc_forward(torch::Tensor input, int64_t bins, double min, double max) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bins > 0, "bins must be positive");

    auto output = torch::zeros({bins}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    histc_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        input.numel(),
        static_cast<int>(bins),
        static_cast<float>(min),
        static_cast<float>(max),
        stream);

    return output;
}

void register_histc(py::module& m) {
    m.def("histc_forward", &histc_forward,
          "Histogram forward",
          py::arg("input"), py::arg("bins"), py::arg("min"), py::arg("max"));
}

REGISTER_BINDING(histc, register_histc);
