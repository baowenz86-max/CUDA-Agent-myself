#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void sigmoid_forward_launcher(const float* x, float* y, int64_t numel, cudaStream_t stream);
extern "C" void hardtanh_forward_launcher(const float* x, float* y, int64_t numel, cudaStream_t stream);

// Sigmoid forward wrapper
torch::Tensor sigmoid_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");

    // Make contiguous if needed
    auto x_contig = x.contiguous();

    // Allocate output tensor of same shape
    auto y = torch::empty_like(x_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    sigmoid_forward_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        x_contig.numel(),
        stream
    );

    return y;
}

// HardTanh forward wrapper
torch::Tensor hardtanh_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");

    // Make contiguous if needed
    auto x_contig = x.contiguous();

    // Allocate output tensor of same shape
    auto y = torch::empty_like(x_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    hardtanh_forward_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        x_contig.numel(),
        stream
    );

    return y;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("sigmoid_forward", &sigmoid_forward, "Sigmoid forward (CUDA)", py::arg("x"));
    m.def("hardtanh_forward", &hardtanh_forward, "HardTanh forward (CUDA)", py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
