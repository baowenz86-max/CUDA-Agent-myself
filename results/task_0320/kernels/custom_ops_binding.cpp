#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void hardtanh_forward_launcher(const float* input,
                                           float* output,
                                           int64_t numel,
                                           cudaStream_t stream);

extern "C" void leaky_relu_forward_launcher(const float* input,
                                             float* output,
                                             int64_t numel,
                                             float negative_slope,
                                             cudaStream_t stream);

// PyTorch wrapper for hardtanh
torch::Tensor hardtanh_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    hardtanh_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );

    return output;
}

// PyTorch wrapper for leaky_relu
torch::Tensor leaky_relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    constexpr float negative_slope = 0.01f;
    leaky_relu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        negative_slope,
        stream
    );

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("hardtanh_forward", &hardtanh_forward, "Hardtanh forward",
          py::arg("x"));
    m.def("leaky_relu_forward", &leaky_relu_forward, "Leaky ReLU forward",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
