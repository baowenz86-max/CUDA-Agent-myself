#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void abs_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream);

extern "C" void add_scalar_tensor_forward_launcher(
    float* output,
    const float* x,
    const float* param,
    int64_t numel,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    cudaStream_t stream);

torch::Tensor abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "abs_forward: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "abs_forward: x must be float32");
    TORCH_CHECK(x.is_contiguous(), "abs_forward: x must be contiguous");
    TORCH_CHECK(x.numel() > 0, "abs_forward: x must have at least one element");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    abs_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        stream);

    return output;
}

torch::Tensor add_scalar_tensor_forward(torch::Tensor x, torch::Tensor param) {
    TORCH_CHECK(x.is_cuda(), "add_scalar_tensor_forward: x must be a CUDA tensor");
    TORCH_CHECK(param.is_cuda(), "add_scalar_tensor_forward: param must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "add_scalar_tensor_forward: x must be float32");
    TORCH_CHECK(param.dtype() == torch::kFloat32, "add_scalar_tensor_forward: param must be float32");
    TORCH_CHECK(x.is_contiguous(), "add_scalar_tensor_forward: x must be contiguous");
    TORCH_CHECK(param.is_contiguous(), "add_scalar_tensor_forward: param must be contiguous");
    TORCH_CHECK(x.numel() > 0, "add_scalar_tensor_forward: x must have at least one element");

    TORCH_CHECK(x.dim() == 4, "add_scalar_tensor_forward: x must be 4D");
    TORCH_CHECK(param.dim() == 3, "add_scalar_tensor_forward: param must be 3D");
    TORCH_CHECK(param.size(0) == x.size(1), "param channels must match x channels");
    TORCH_CHECK(param.size(1) == x.size(2) && param.size(2) == x.size(3),
                "param spatial dims must match x spatial dims");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t batch = x.size(0);
    int64_t channels = x.size(1);
    int64_t height = x.size(2);
    int64_t width = x.size(3);

    add_scalar_tensor_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        param.data_ptr<float>(),
        x.numel(),
        batch,
        channels,
        height,
        width,
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward, "Element-wise absolute value", py::arg("x"));
    m.def("add_scalar_tensor_forward", &add_scalar_tensor_forward,
          "Broadcast addition of param over batch dimension",
          py::arg("x"), py::arg("param"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
