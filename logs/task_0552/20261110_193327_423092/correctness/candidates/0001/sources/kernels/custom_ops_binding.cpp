#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void elu_forward_launcher(const float* x, float* y, int64_t n, cudaStream_t stream);
extern "C" void abs_forward_launcher(const float* x, float* y, int64_t n, cudaStream_t stream);
extern "C" void elu_abs_forward_launcher(const float* x, float* y, int64_t n, cudaStream_t stream);

// Helper to validate and get contiguous float32 CUDA tensor
static void validate_input(const torch::Tensor& x, const char* op_name) {
    TORCH_CHECK(x.is_cuda(), op_name, ": input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, op_name, ": input must be float32");
    TORCH_CHECK(x.is_contiguous(), op_name, ": input must be contiguous");
}

// Wrapper for elu_forward
torch::Tensor elu_forward(torch::Tensor x) {
    validate_input(x, "elu_forward");
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    elu_forward_launcher(x.data_ptr<float>(), out.data_ptr<float>(), n, stream);
    return out;
}

// Wrapper for abs_forward
torch::Tensor abs_forward(torch::Tensor x) {
    validate_input(x, "abs_forward");
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    abs_forward_launcher(x.data_ptr<float>(), out.data_ptr<float>(), n, stream);
    return out;
}

// Wrapper for elu_abs_forward
torch::Tensor elu_abs_forward(torch::Tensor x) {
    validate_input(x, "elu_abs_forward");
    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    elu_abs_forward_launcher(x.data_ptr<float>(), out.data_ptr<float>(), n, stream);
    return out;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("elu_forward", &elu_forward, "ELU activation forward", py::arg("x"));
    m.def("abs_forward", &abs_forward, "Absolute value forward", py::arg("x"));
    m.def("elu_abs_forward", &elu_abs_forward, "Fused ELU + absolute value forward", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
