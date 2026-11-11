#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Launcher declarations
extern "C" void batchnorm_inference_launcher(
    const float* x,
    float* out,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N,
    int C,
    int HW,
    cudaStream_t stream
);

extern "C" void triplet_margin_loss_sum_launcher(
    const float* anchor,
    const float* positive,
    const float* negative,
    float* out,
    float margin,
    int N,
    int D,
    cudaStream_t stream
);

torch::Tensor batchnorm_inference(
    torch::Tensor x,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    if (weight.has_value()) {
        TORCH_CHECK(weight.value().is_cuda(), "weight must be a CUDA tensor if provided");
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor if provided");
    }

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");
    if (weight.has_value()) {
        TORCH_CHECK(weight.value().dtype() == torch::kFloat32, "weight must be float32 if provided");
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "bias must be float32 if provided");
    }

    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t HW = H * W;

    TORCH_CHECK(running_mean.dim() == 1 && running_mean.size(0) == C, "running_mean must be 1D of shape [C]");
    TORCH_CHECK(running_var.dim() == 1 && running_var.size(0) == C, "running_var must be 1D of shape [C]");
    if (weight.has_value()) {
        TORCH_CHECK(weight.value().dim() == 1 && weight.value().size(0) == C, "weight must be 1D of shape [C] if provided");
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().dim() == 1 && bias.value().size(0) == C, "bias must be 1D of shape [C] if provided");
    }

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");
    if (weight.has_value()) {
        TORCH_CHECK(weight.value().is_contiguous(), "weight must be contiguous if provided");
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous if provided");
    }

    auto out = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* weight_ptr = weight.has_value() ? weight.value().data_ptr<float>() : nullptr;
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    batchnorm_inference_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(HW),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

torch::Tensor triplet_margin_loss_sum(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    double margin,
    int64_t p)
{
    TORCH_CHECK(anchor.is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(positive.is_cuda(), "positive must be a CUDA tensor");
    TORCH_CHECK(negative.is_cuda(), "negative must be a CUDA tensor");

    TORCH_CHECK(anchor.dtype() == torch::kFloat32, "anchor must be float32");
    TORCH_CHECK(positive.dtype() == torch::kFloat32, "positive must be float32");
    TORCH_CHECK(negative.dtype() == torch::kFloat32, "negative must be float32");

    TORCH_CHECK(positive.sizes() == anchor.sizes(), "positive must have same shape as anchor");
    TORCH_CHECK(negative.sizes() == anchor.sizes(), "negative must have same shape as anchor");

    TORCH_CHECK(p == 2, "Only p=2 is supported in this wrapper");

    TORCH_CHECK(anchor.is_contiguous(), "anchor must be contiguous");
    TORCH_CHECK(positive.is_contiguous(), "positive must be contiguous");
    TORCH_CHECK(negative.is_contiguous(), "negative must be contiguous");

    int64_t N = anchor.size(0);
    int64_t D = anchor.numel() / N;

    auto out = torch::empty({}, anchor.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    triplet_margin_loss_sum_launcher(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<float>(margin),
        static_cast<int>(N),
        static_cast<int>(D),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("batchnorm_inference", &batchnorm_inference,
          "BatchNorm2d inference forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps"));

    m.def("triplet_margin_loss_sum", &triplet_margin_loss_sum,
          "TripletMarginLoss sum forward",
          py::arg("anchor"),
          py::arg("positive"),
          py::arg("negative"),
          py::arg("margin"),
          py::arg("p"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
