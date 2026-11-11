#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void batch_norm_eval_launcher(
    const float* x,
    float* out,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int N,
    int C,
    int H,
    int W,
    cudaStream_t stream);

// Wrapper for batch_norm_eval_forward
torch::Tensor batch_norm_eval_forward(
    torch::Tensor x,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps = 1e-05) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must have 4 dimensions [N, C, H, W]");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Validate running_mean and running_var
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_mean.numel() == C, "running_mean must have C elements");

    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");
    TORCH_CHECK(running_var.numel() == C, "running_var must have C elements");

    // Validate optional weight and bias
    const float* weight_ptr = nullptr;
    if (weight.has_value()) {
        auto w = weight.value();
        TORCH_CHECK(w.is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(w.dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(w.is_contiguous(), "weight must be contiguous");
        TORCH_CHECK(w.numel() == C, "weight must have C elements");
        weight_ptr = w.data_ptr<float>();
    }

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(b.numel() == C, "bias must have C elements");
        bias_ptr = b.data_ptr<float>();
    }

    // Allocate output tensor
    auto out = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    batch_norm_eval_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        N,
        C,
        H,
        W,
        stream);

    return out;
}

// Registration function
void register_batch_norm_eval(pybind11::module& m) {
    m.def("batch_norm_eval_forward", &batch_norm_eval_forward,
          "BatchNorm2d evaluation forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps") = 1e-05);
}

REGISTER_BINDING(batch_norm_eval, register_batch_norm_eval);
