#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void clamp_forward_launcher(
    float* out,
    const float* in,
    int N, int C, int H, int W,
    int64_t out_N_stride, int64_t out_C_stride, int64_t out_H_stride, int64_t out_W_stride,
    int64_t in_N_stride, int64_t in_C_stride, int64_t in_H_stride, int64_t in_W_stride,
    float min_val, float max_val,
    cudaStream_t stream);

extern "C" void bn2d_eval_forward_launcher(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N, int C, int H, int W,
    int64_t out_N_stride, int64_t out_C_stride, int64_t out_H_stride, int64_t out_W_stride,
    int64_t in_N_stride, int64_t in_C_stride, int64_t in_H_stride, int64_t in_W_stride,
    cudaStream_t stream);

extern "C" void clamp_bn2d_forward_launcher(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    float clamp_min, float clamp_max,
    int N, int C, int H, int W,
    int64_t out_N_stride, int64_t out_C_stride, int64_t out_H_stride, int64_t out_W_stride,
    int64_t in_N_stride, int64_t in_C_stride, int64_t in_H_stride, int64_t in_W_stride,
    cudaStream_t stream);

namespace {

// Helper to extract 4D dims and strides, ensuring contiguity via clone if needed?
// We allow arbitrary strides, so we pass raw strides.
void validate_4d_cuda_float(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(x.dim() == 4, name, " must be 4D (N,C,H,W)");
}

void validate_channel_tensor(const torch::Tensor& t, const char* name, bool nullable) {
    if (nullable && !t.defined()) return;
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.dim() == 1, name, " must be 1D (C)");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

} // anonymous namespace

torch::Tensor clamp_forward(torch::Tensor x, double min_val, double max_val) {
    validate_4d_cuda_float(x, "x");
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto out = torch::empty_like(x); // same shape, strides, dtype, device

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    clamp_forward_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        static_cast<float>(min_val), static_cast<float>(max_val),
        stream);

    return out;
}

torch::Tensor bn2d_eval_forward(torch::Tensor x, c10::optional<torch::Tensor> weight, c10::optional<torch::Tensor> bias, torch::Tensor running_mean, torch::Tensor running_var, double eps) {
    validate_4d_cuda_float(x, "x");
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    TORCH_CHECK(running_mean.is_cuda() && running_mean.dtype() == torch::kFloat32 && running_mean.dim() == 1 && running_mean.is_contiguous(), "running_mean must be contiguous float32 CUDA 1D tensor");
    TORCH_CHECK(running_var.is_cuda() && running_var.dtype() == torch::kFloat32 && running_var.dim() == 1 && running_var.is_contiguous(), "running_var must be contiguous float32 CUDA 1D tensor");
    TORCH_CHECK(running_mean.size(0) == C, "running_mean size must match C");
    TORCH_CHECK(running_var.size(0) == C, "running_var size must match C");

    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    if (weight.has_value() && weight->defined()) {
        TORCH_CHECK(weight->is_cuda() && weight->dtype() == torch::kFloat32 && weight->dim() == 1 && weight->is_contiguous(), "weight must be contiguous float32 CUDA 1D tensor");
        TORCH_CHECK(weight->size(0) == C, "weight size must match C");
        weight_ptr = weight->data_ptr<float>();
    }
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->is_cuda() && bias->dtype() == torch::kFloat32 && bias->dim() == 1 && bias->is_contiguous(), "bias must be contiguous float32 CUDA 1D tensor");
        TORCH_CHECK(bias->size(0) == C, "bias size must match C");
        bias_ptr = bias->data_ptr<float>();
    }

    auto out = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bn2d_eval_forward_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        N, C, H, W,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream);

    return out;
}

torch::Tensor clamp_bn2d_forward(torch::Tensor x, c10::optional<torch::Tensor> weight, c10::optional<torch::Tensor> bias, torch::Tensor running_mean, torch::Tensor running_var, double eps) {
    // Same validations as bn2d_eval_forward, but with clamp values fixed to 0 and 1
    validate_4d_cuda_float(x, "x");
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    TORCH_CHECK(running_mean.is_cuda() && running_mean.dtype() == torch::kFloat32 && running_mean.dim() == 1 && running_mean.is_contiguous(), "running_mean must be contiguous float32 CUDA 1D tensor");
    TORCH_CHECK(running_var.is_cuda() && running_var.dtype() == torch::kFloat32 && running_var.dim() == 1 && running_var.is_contiguous(), "running_var must be contiguous float32 CUDA 1D tensor");
    TORCH_CHECK(running_mean.size(0) == C, "running_mean size must match C");
    TORCH_CHECK(running_var.size(0) == C, "running_var size must match C");

    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    if (weight.has_value() && weight->defined()) {
        TORCH_CHECK(weight->is_cuda() && weight->dtype() == torch::kFloat32 && weight->dim() == 1 && weight->is_contiguous(), "weight must be contiguous float32 CUDA 1D tensor");
        TORCH_CHECK(weight->size(0) == C, "weight size must match C");
        weight_ptr = weight->data_ptr<float>();
    }
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->is_cuda() && bias->dtype() == torch::kFloat32 && bias->dim() == 1 && bias->is_contiguous(), "bias must be contiguous float32 CUDA 1D tensor");
        TORCH_CHECK(bias->size(0) == C, "bias size must match C");
        bias_ptr = bias->data_ptr<float>();
    }

    auto out = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    clamp_bn2d_forward_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        0.0f, 1.0f,
        N, C, H, W,
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream);

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("clamp_forward", &clamp_forward, "Clamp forward", py::arg("x"), py::arg("min_val"), py::arg("max_val"));
    m.def("bn2d_eval_forward", &bn2d_eval_forward, "BatchNorm2d eval forward", py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("running_mean"), py::arg("running_var"), py::arg("eps"));
    m.def("clamp_bn2d_forward", &clamp_bn2d_forward, "Fused clamp and BatchNorm2d eval forward", py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("running_mean"), py::arg("running_var"), py::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
