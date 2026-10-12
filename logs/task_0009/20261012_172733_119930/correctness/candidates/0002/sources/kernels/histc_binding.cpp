#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration, defined in kernels/histc.cu
extern "C" void histc_forward_launcher(
    const float* input,
    float* output,
    int num_elements,
    int bins,
    float min,
    float max,
    cudaStream_t stream);

// Binding function: computes histogram of a CUDA float tensor
// Input: any contiguous float CUDA tensor
// bins: number of histogram bins
// min, max: range for histogram
static torch::Tensor histc_forward(torch::Tensor input, int64_t bins, double min, double max) {
    // Parameter validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(bins > 0, "bins must be positive");

    auto output = torch::zeros({bins}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    histc_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(input.numel()),
        static_cast<int>(bins),
        static_cast<float>(min),
        static_cast<float>(max),
        stream
    );

    return output;
}

void register_histc(pybind11::module& m) {
    m.def("histc_forward", &histc_forward,
          "Histogram forward (CUDA)",
          pybind11::arg("input"), pybind11::arg("bins"), pybind11::arg("min"), pybind11::arg("max"));
}

REGISTER_BINDING(histc, register_histc);
