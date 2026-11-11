// kernels/custom_ops_binding.cpp
// PyTorch binding for custom CUDA ops: relu_forward and batch_norm1d_eval_forward
// Contains only tensor validation, launcher invocation, and pybind registration.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream);

extern "C" void batch_norm1d_eval_launcher(
    float* output,
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t N,
    int64_t C,
    int64_t L,
    cudaStream_t stream);

// ReLU forward wrapper
torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "relu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "relu_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "relu_forward: input must be contiguous");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    relu_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        stream);

    return output;
}

// BatchNorm1d eval forward wrapper
torch::Tensor batch_norm1d_eval_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {

    TORCH_CHECK(x.is_cuda(), "batch_norm1d_eval_forward: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "batch_norm1d_eval_forward: x must be float32");
    TORCH_CHECK(x.is_contiguous(), "batch_norm1d_eval_forward: x must be contiguous");

    // Check other tensors
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be CUDA tensor");

    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");

    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");

    // Shape checks: x must be 3D [N, C, L]; other tensors must be [C]
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N, C, L)");
    int64_t C = x.size(1);
    TORCH_CHECK(weight.numel() == C, "weight shape must match C");
    TORCH_CHECK(bias.numel() == C, "bias shape must match C");
    TORCH_CHECK(running_mean.numel() == C, "running_mean shape must match C");
    TORCH_CHECK(running_var.numel() == C, "running_var shape must match C");

    auto output = torch::empty_like(x);
    int64_t N = x.size(0);
    int64_t L = x.size(2);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batch_norm1d_eval_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        N,
        C,
        L,
        stream);

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("relu_forward", &relu_forward, "ReLU forward", py::arg("x"));
    m.def("batch_norm1d_eval_forward", &batch_norm1d_eval_forward,
          "BatchNorm1d eval forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
