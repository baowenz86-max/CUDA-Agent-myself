// Binding for abs_forward and gt_scalar_forward
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void abs_forward_launcher(const float* input,
                                     float* output,
                                     size_t total_elements,
                                     cudaStream_t stream);

extern "C" void gt_scalar_forward_launcher(const float* input,
                                           bool* output,
                                           float threshold,
                                           size_t total_elements,
                                           cudaStream_t stream);

// Wrapper for abs_forward
torch::Tensor abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);
    size_t total_elements = static_cast<size_t>(x.numel());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    abs_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        stream
    );

    return output;
}

// Wrapper for gt_scalar_forward
torch::Tensor gt_scalar_forward(torch::Tensor x, double threshold) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty(x.sizes(), x.options().dtype(torch::kBool));
    size_t total_elements = static_cast<size_t>(x.numel());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gt_scalar_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<bool>(),
        static_cast<float>(threshold),
        total_elements,
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward,
          "Element-wise absolute value",
          py::arg("x"));
    m.def("gt_scalar_forward", &gt_scalar_forward,
          "Element-wise greater-than scalar comparison",
          py::arg("x"),
          py::arg("threshold") = 0.5);
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
