#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from .cu file
extern "C" void histc_custom_launcher(
    const float* input,
    float* output,
    int num_elements,
    int bins,
    float min_val,
    float max_val,
    cudaStream_t stream
);

// PyTorch binding function
torch::Tensor histc_custom_forward(
    torch::Tensor input,
    double start,
    double end,
    int64_t bins)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(bins > 0, "bins must be positive");
    TORCH_CHECK(start < end, "start must be less than end");

    // Allocate output histogram tensor
    auto output = torch::zeros({bins}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call CUDA launcher
    histc_custom_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(input.numel()),
        static_cast<int>(bins),
        static_cast<float>(start),
        static_cast<float>(end),
        stream
    );

    return output;
}

// Registration function
void register_histc_custom(pybind11::module& m) {
    m.def("histc_custom_forward", &histc_custom_forward,
          "Custom histogram forward (CUDA)",
          py::arg("input"),
          py::arg("start"),
          py::arg("end"),
          py::arg("bins"));
}

// Auto-register
REGISTER_BINDING(histc_custom, register_histc_custom);
