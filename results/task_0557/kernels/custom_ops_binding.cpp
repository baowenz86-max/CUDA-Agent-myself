#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void mish_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream
);

// PyTorch tensor wrapper
torch::Tensor mish_forward(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");

    // Allocate output with same shape/device/dtype
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    mish_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );

    return output;
}

// Registration function
void register_mish_forward(pybind11::module& m) {
    m.def("mish_forward", &mish_forward,
          "Mish activation forward",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(mish_forward, register_mish_forward);
