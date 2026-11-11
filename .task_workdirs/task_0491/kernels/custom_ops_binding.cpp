#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declarations from custom_ops.cu
extern "C" void batch_norm_2d_eval_launcher(
    float* y,
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream
);

extern "C" void clamp_forward_launcher(
    float* y,
    const float* x,
    double min_val,
    double max_val,
    int64_t numel,
    cudaStream_t stream
);

// PyTorch wrapper for batch_norm_2d_eval
torch::Tensor batch_norm_2d_eval(
    torch::Tensor x,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Optional weight and bias validation
    float* weight_ptr = nullptr;
    float* bias_ptr = nullptr;
    if (weight.has_value()) {
        auto w = weight.value();
        TORCH_CHECK(w.is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(w.dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(w.numel() == C, "weight must have C elements");
        TORCH_CHECK(w.is_contiguous(), "weight must be contiguous");
        weight_ptr = w.data_ptr<float>();
    }
    if (bias.has_value()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.numel() == C, "bias must have C elements");
        TORCH_CHECK(b.is_contiguous(), "bias must be contiguous");
        bias_ptr = b.data_ptr<float>();
    }

    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");
    TORCH_CHECK(running_mean.numel() == C, "running_mean must have C elements");
    TORCH_CHECK(running_var.numel() == C, "running_var must have C elements");

    // Allocate output tensor with same shape and options as input
    auto y = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    batch_norm_2d_eval_launcher(
        y.data_ptr<float>(),
        x.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        N,
        C,
        H,
        W,
        stream
    );

    // Check for errors after launch (optional but recommended)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("batch_norm_2d_eval CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return y;
}

// PyTorch wrapper for clamp_forward
torch::Tensor clamp_forward(
    torch::Tensor x,
    double min_val,
    double max_val
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    int64_t numel = x.numel();
    auto y = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    clamp_forward_launcher(
        y.data_ptr<float>(),
        x.data_ptr<float>(),
        min_val,
        max_val,
        numel,
        stream
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("clamp_forward CUDA kernel launch failed: ", cudaGetErrorString(err));
    }

    return y;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm_2d_eval", &batch_norm_2d_eval,
          "BatchNorm2d evaluation mode",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps"));
    m.def("clamp_forward", &clamp_forward,
          "Clamp forward",
          py::arg("x"),
          py::arg("min_val"),
          py::arg("max_val"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
