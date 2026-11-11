#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of the CUDA launcher from custom_ops.cu
extern "C" void hardtanh_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    float min_val,
    float max_val,
    cudaStream_t stream);

// Wrapper function for the Python-visible operator
torch::Tensor hardtanh_forward(torch::Tensor x, double min_val, double max_val) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");

    // Ensure input is contiguous; elementwise op handles arbitrary strides by
    // making it contiguous. This is safe and avoids host/device pointer issues.
    x = x.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto output = torch::empty_like(x); // preserves contiguous layout

    int64_t numel = x.numel();

    hardtanh_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_hardtanh(pybind11::module& m) {
    m.def("hardtanh_forward", &hardtanh_forward,
          "Hardtanh forward (element-wise clamp)",
          py::arg("x"),
          py::arg("min_val") = -1.0,
          py::arg("max_val") = 1.0);
}

// Auto-register
REGISTER_BINDING(hardtanh, register_hardtanh);
