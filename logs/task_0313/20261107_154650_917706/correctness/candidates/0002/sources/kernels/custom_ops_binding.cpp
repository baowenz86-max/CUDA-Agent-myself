#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// extern "C" declarations matching custom_ops.cu
extern "C" {
    void batchnorm2d_eval_forward(
        float* output,
        const float* input,
        const float* running_mean,
        const float* running_var,
        const float* weight,
        const float* bias,
        float eps,
        int B,
        int C,
        int H,
        int W,
        cudaStream_t stream);

    void prod_channel_forward(
        float* output,
        const float* input,
        int B,
        int C,
        int H,
        int W,
        cudaStream_t stream);
}

static void check_batchnorm_inputs(
    const torch::Tensor& input,
    const torch::Tensor& running_mean,
    const torch::Tensor& running_var,
    const c10::optional<torch::Tensor>& weight,
    const c10::optional<torch::Tensor>& bias)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (B,C,H,W)");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be CUDA tensor");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_mean.dim() == 1, "running_mean must be 1D");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_mean.size(0) == input.size(1), "running_mean size must equal C");

    TORCH_CHECK(running_var.is_cuda(), "running_var must be CUDA tensor");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");
    TORCH_CHECK(running_var.dim() == 1, "running_var must be 1D");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");
    TORCH_CHECK(running_var.size(0) == input.size(1), "running_var size must equal C");

    if (weight.has_value()) {
        TORCH_CHECK(weight.value().is_cuda(), "weight must be CUDA tensor");
        TORCH_CHECK(weight.value().dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight.value().dim() == 1, "weight must be 1D");
        TORCH_CHECK(weight.value().is_contiguous(), "weight must be contiguous");
        TORCH_CHECK(weight.value().size(0) == input.size(1), "weight size must equal C");
    }
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be CUDA tensor");
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias.value().is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.value().size(0) == input.size(1), "bias size must equal C");
    }
}

static void check_prod_inputs(const torch::Tensor& input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (B,C,H,W)");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(dim == 1, "prod_channel_forward only supports dim=1");
}

torch::Tensor batchnorm2d_eval_forward_binding(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    float eps)
{
    check_batchnorm_inputs(input, running_mean, running_var, weight, bias);

    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    auto output = torch::empty_like(input);

    const float* weight_ptr = weight.has_value() ? weight.value().data_ptr<float>() : nullptr;
    const float* bias_ptr = bias.has_value() ? bias.value().data_ptr<float>() : nullptr;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    batchnorm2d_eval_forward(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        eps,
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream);

    return output;
}

torch::Tensor prod_channel_forward_binding(torch::Tensor input, int64_t dim) {
    check_prod_inputs(input, dim);

    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    auto output = torch::empty({B, 1, H, W}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    prod_channel_forward(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("batchnorm2d_eval_forward", &batchnorm2d_eval_forward_binding,
          "BatchNorm2d eval forward",
          py::arg("input"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("weight") = c10::nullopt,
          py::arg("bias") = c10::nullopt,
          py::arg("eps"));

    m.def("prod_channel_forward", &prod_channel_forward_binding,
          "Product along channel dimension (dim=1, keepdim=True)",
          py::arg("input"),
          py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);