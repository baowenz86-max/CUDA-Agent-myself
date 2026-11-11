#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu.
extern "C" void norm_forward_launcher(
    const float* x,
    float* out,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    const int64_t* x_strides,
    cudaStream_t stream);

extern "C" void norm_forward_launcher_strides(
    const float* x,
    float* out,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t stride_N,
    int64_t stride_C,
    int64_t stride_H,
    int64_t stride_W,
    cudaStream_t stream);

static const float* optional_data_ptr(const c10::optional<torch::Tensor>& t) {
    if (t.has_value() && t->defined()) {
        return t->data_ptr<float>();
    }
    return nullptr;
}

torch::Tensor norm_forward(
    torch::Tensor x,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> running_mean,
    c10::optional<torch::Tensor> running_var,
    double eps,
    double momentum,
    bool training)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [N, C, H, W]");

    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);
    TORCH_CHECK(C > 0, "C must be positive");

    if (weight.has_value() && weight->defined()) {
        TORCH_CHECK(weight->is_cuda(), "weight must be CUDA");
        TORCH_CHECK(weight->dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight->numel() == C, "weight must have size [C]");
        TORCH_CHECK(weight->is_contiguous(), "weight must be contiguous");
    }
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias->dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias->numel() == C, "bias must have size [C]");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
    }
    if (running_mean.has_value() && running_mean->defined()) {
        TORCH_CHECK(running_mean->is_cuda(), "running_mean must be CUDA");
        TORCH_CHECK(running_mean->dtype() == torch::kFloat32, "running_mean must be float32");
        TORCH_CHECK(running_mean->numel() == C, "running_mean must have size [C]");
        TORCH_CHECK(running_mean->is_contiguous(), "running_mean must be contiguous");
    }
    if (running_var.has_value() && running_var->defined()) {
        TORCH_CHECK(running_var->is_cuda(), "running_var must be CUDA");
        TORCH_CHECK(running_var->dtype() == torch::kFloat32, "running_var must be float32");
        TORCH_CHECK(running_var->numel() == C, "running_var must have size [C]");
        TORCH_CHECK(running_var->is_contiguous(), "running_var must be contiguous");
    }

    auto out = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    norm_forward_launcher_strides(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        optional_data_ptr(weight),
        optional_data_ptr(bias),
        optional_data_ptr(running_mean),
        optional_data_ptr(running_var),
        static_cast<float>(eps),
        N, C, H, W,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "norm_forward kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

void register_norm_forward(pybind11::module& m) {
    m.def("norm_forward", &norm_forward,
          "BatchNorm2d evaluation forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps") = 1e-05,
          py::arg("momentum") = 0.1,
          py::arg("training") = false);
}

REGISTER_BINDING(norm_forward, register_norm_forward);
