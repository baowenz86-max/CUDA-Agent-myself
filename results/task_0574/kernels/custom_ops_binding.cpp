// PyTorch bindings for custom CUDA kernels: isnan and BatchNorm1d forward
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void isnan_forward_launcher(
    const float* input,
    uint8_t* output,
    int64_t numel,
    cudaStream_t stream);

extern "C" void batch_norm_forward_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    float momentum,
    int64_t N,
    int64_t C,
    int64_t L,
    float* out,
    cudaStream_t stream);

// Element-wise isnan forward
torch::Tensor isnan_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto out = torch::empty_like(x, x.options().dtype(torch::kBool));
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    isnan_forward_launcher(
        x.data_ptr<float>(),
        reinterpret_cast<uint8_t*>(out.data_ptr<bool>()),
        numel,
        stream);

    return out;
}

// BatchNorm1d forward (eval mode only as per contract)
torch::Tensor batch_norm_forward(
    torch::Tensor x,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> running_mean,
    c10::optional<torch::Tensor> running_var,
    bool training,
    double eps,
    double momentum) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N, C, L)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t L = x.size(2);

    auto check_optional_channel_tensor = [C](const c10::optional<torch::Tensor>& t, const char* name) {
        if (t.has_value()) {
            TORCH_CHECK(t->is_cuda(), name, " must be a CUDA tensor");
            TORCH_CHECK(t->scalar_type() == torch::kFloat32, name, " must be float32");
            TORCH_CHECK(t->is_contiguous(), name, " must be contiguous");
            TORCH_CHECK(t->numel() == C, name, " must have numel equal to C");
        }
    };

    check_optional_channel_tensor(weight, "weight");
    check_optional_channel_tensor(bias, "bias");
    check_optional_channel_tensor(running_mean, "running_mean");
    check_optional_channel_tensor(running_var, "running_var");

    // In eval mode, running stats are required. In training mode, they may be null but this kernel does not support training.
    if (!training) {
        TORCH_CHECK(running_mean.has_value(), "running_mean is required in eval mode");
        TORCH_CHECK(running_var.has_value(), "running_var is required in eval mode");
    }

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    batch_norm_forward_launcher(
        x.data_ptr<float>(),
        weight.has_value() ? weight->data_ptr<float>() : nullptr,
        bias.has_value() ? bias->data_ptr<float>() : nullptr,
        running_mean.has_value() ? running_mean->data_ptr<float>() : nullptr,
        running_var.has_value() ? running_var->data_ptr<float>() : nullptr,
        static_cast<float>(eps),
        static_cast<float>(momentum),
        N,
        C,
        L,
        out.data_ptr<float>(),
        stream);

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("isnan_forward", &isnan_forward,
          "Element-wise isnan forward",
          py::arg("x"));

    m.def("batch_norm_forward", &batch_norm_forward,
          "BatchNorm1d forward (eval mode)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("training"),
          py::arg("eps"),
          py::arg("momentum") = 0.1);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
