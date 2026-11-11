#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void hardshrink_launcher(
    float* output,
    const float* input,
    int64_t N,
    cudaStream_t stream
);

// PyTorch wrapper: hardshrink elementwise with lambda=0.5
torch::Tensor hardshrink_forward(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call CUDA launcher
    hardshrink_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        stream
    );

    return output;
}

// Registration function
void register_hardshrink(pybind11::module& m) {
    m.def("hardshrink_forward", &hardshrink_forward,
          "Hardshrink forward with lambda=0.5",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(hardshrink, register_hardshrink);
