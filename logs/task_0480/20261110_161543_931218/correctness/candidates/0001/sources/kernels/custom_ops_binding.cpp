#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu_forward_launcher(const float* input, float* output, int64_t n, cudaStream_t stream);
extern "C" void softplus_forward_launcher(const float* input, float* output, int64_t n, cudaStream_t stream);

// Wrapper for relu_forward
torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    relu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int64_t>(x.numel()),
        stream
    );

    return output;
}

// Wrapper for softplus_forward
torch::Tensor softplus_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softplus_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int64_t>(x.numel()),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("relu_forward", &relu_forward, "ReLU forward", py::arg("x"));
    m.def("softplus_forward", &softplus_forward, "Softplus forward", py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
