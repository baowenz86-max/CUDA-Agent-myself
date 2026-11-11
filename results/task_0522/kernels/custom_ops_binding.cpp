#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void batch_norm_eval_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N,
    int C,
    int H,
    int W,
    cudaStream_t stream
);

static torch::Tensor batch_norm_eval_forward(
    torch::Tensor x,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps
) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");

    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_mean.dim() == 1, "running_mean must be 1D");

    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");
    TORCH_CHECK(running_var.dim() == 1, "running_var must be 1D");

    int C = x.size(1);
    TORCH_CHECK(running_mean.numel() == C, "running_mean size must match channels");
    TORCH_CHECK(running_var.numel() == C, "running_var size must match channels");

    const float* weight_ptr = nullptr;
    if (weight.has_value()) {
        auto w = weight.value();
        TORCH_CHECK(w.is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(w.is_contiguous(), "weight must be contiguous");
        TORCH_CHECK(w.dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(w.dim() == 1 && w.numel() == C, "weight size must match channels");
        weight_ptr = w.data_ptr<float>();
    }

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(b.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.dim() == 1 && b.numel() == C, "bias size must match channels");
        bias_ptr = b.data_ptr<float>();
    }

    int N = x.size(0);
    int H = x.size(2);
    int W = x.size(3);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batch_norm_eval_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        N,
        C,
        H,
        W,
        stream
    );

    return output;
}

void register_batch_norm_eval(pybind11::module& m) {
    m.def("batch_norm_eval", &batch_norm_eval_forward,
          "BatchNorm2d eval forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps"));
}

REGISTER_BINDING(batch_norm_eval, register_batch_norm_eval);
