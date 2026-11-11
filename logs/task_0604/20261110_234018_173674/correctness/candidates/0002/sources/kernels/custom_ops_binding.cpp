#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers with new simplified signature
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

void validate_and_launch(
    torch::Tensor x,
    torch::Tensor& output,
    const std::function<void(const float*, float*, int64_t, cudaStream_t)>& launcher)
{
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

} // namespace

torch::Tensor abs_forward(torch::Tensor x) {
    torch::Tensor output;
    validate_and_launch(x, output, [](const float* in, float* out, int64_t n, cudaStream_t s) {
        abs_forward_launcher(in, out, n, s);
    });
    return output;
}

torch::Tensor relu6_forward(torch::Tensor x) {
    torch::Tensor output;
    validate_and_launch(x, output, [](const float* in, float* out, int64_t n, cudaStream_t s) {
        relu6_forward_launcher(in, out, n, s);
    });
    return output;
}

torch::Tensor abs_relu6_forward(torch::Tensor x) {
    torch::Tensor output;
    validate_and_launch(x, output, [](const float* in, float* out, int64_t n, cudaStream_t s) {
        abs_relu6_forward_launcher(in, out, n, s);
    });
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("abs_forward", &abs_forward, "Element-wise absolute value", py::arg("x"));
    m.def("relu6_forward", &relu6_forward, "Element-wise ReLU6", py::arg("x"));
    m.def("abs_relu6_forward", &abs_relu6_forward, "Fused abs then ReLU6", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
