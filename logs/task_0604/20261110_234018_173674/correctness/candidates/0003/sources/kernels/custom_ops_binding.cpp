#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void abs_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream);

extern "C" void relu6_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream);

extern "C" void abs_relu6_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream);

namespace {

// Generic launch wrapper for elementwise ops on contiguous tensors
void launch_elementwise(
    const torch::Tensor& x,
    torch::Tensor& output,
    void (*launcher)(const float*, float*, int64_t, cudaStream_t))
{
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    output = torch::empty_like(x);

    if (x.numel() == 0) {
        return;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream);
}
}

// PyTorch wrapper functions

torch::Tensor abs_forward(torch::Tensor x) {
    torch::Tensor output;
    launch_elementwise(x, output, abs_forward_launcher);
    return output;
}

torch::Tensor relu6_forward(torch::Tensor x) {
    torch::Tensor output;
    launch_elementwise(x, output, relu6_forward_launcher);
    return output;
}

torch::Tensor abs_relu6_forward(torch::Tensor x) {
    torch::Tensor output;
    launch_elementwise(x, output, abs_relu6_forward_launcher);
    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward, "Element-wise absolute value", py::arg("x"));
    m.def("relu6_forward", &relu6_forward, "Element-wise ReLU6", py::arg("x"));
    m.def("abs_relu6_forward", &abs_relu6_forward, "Fused abs then ReLU6", py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
