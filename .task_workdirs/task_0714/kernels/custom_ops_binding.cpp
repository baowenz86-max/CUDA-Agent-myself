#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void abs_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void softplus_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);

torch::Tensor abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    abs_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );
    return output;
}

torch::Tensor softplus_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softplus_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward, "Elementwise absolute value",
          py::arg("x"));
    m.def("softplus_forward", &softplus_forward, "Elementwise softplus",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
