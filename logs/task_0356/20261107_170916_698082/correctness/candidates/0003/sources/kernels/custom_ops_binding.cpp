#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void batch_norm_inference_launcher(
    float* y,
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void group_norm_inference_launcher(
    float* y,
    const float* x,
    const float* weight,
    const float* bias,
    int num_groups,
    float eps,
    int N, int C, int H, int W,
    cudaStream_t stream);

// Helper to validate tensor and return pointer (or nullptr if nullable and not defined)
const float* optional_data_ptr(const torch::Tensor& t) {
    if (!t.defined()) {
        return nullptr;
    }
    return t.data_ptr<float>();
}

torch::Tensor batch_norm_inference(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    float eps) {
    // Validate inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.numel() == C, "weight size must match C");

    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.numel() == C, "bias size must match C");

    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_mean.numel() == C, "running_mean size must match C");

    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");
    TORCH_CHECK(running_var.numel() == C, "running_var size must match C");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batch_norm_inference_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        optional_data_ptr(weight),
        optional_data_ptr(bias),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream);

    return output;
}

torch::Tensor group_norm_inference(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    float eps) {
    // Validate inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");

    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.numel() == C, "weight size must match C");

    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.numel() == C, "bias size must match C");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    group_norm_inference_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        optional_data_ptr(weight),
        optional_data_ptr(bias),
        static_cast<int>(num_groups),
        eps,
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm_inference", &batch_norm_inference,
          "BatchNorm2d inference",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps"));
    m.def("group_norm_inference", &group_norm_inference,
          "GroupNorm inference",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("num_groups"),
          py::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
