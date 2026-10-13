#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void batch_norm_launcher(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t HW,
    cudaStream_t stream);

extern "C" void group_norm_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t HW,
    int64_t groups,
    float eps,
    cudaStream_t stream);

extern "C" void histc_launcher(
    const float* input,
    int64_t numel,
    float min,
    float max,
    int64_t bins,
    float* output,
    cudaStream_t stream);

// Helper to check common tensor requirements
void check_cuda_float_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

torch::Tensor batch_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {

    check_cuda_float_contiguous(input, "input");
    check_cuda_float_contiguous(weight, "weight");
    check_cuda_float_contiguous(bias, "bias");
    check_cuda_float_contiguous(running_mean, "running_mean");
    check_cuda_float_contiguous(running_var, "running_var");

    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t HW = H * W;

    TORCH_CHECK(weight.numel() == C, "weight size must equal channels");
    TORCH_CHECK(bias.numel() == C, "bias size must equal channels");
    TORCH_CHECK(running_mean.numel() == C, "running_mean size must equal channels");
    TORCH_CHECK(running_var.numel() == C, "running_var size must equal channels");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batch_norm_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        N, C, HW,
        stream);

    return output;
}

torch::Tensor group_norm_forward(
    torch::Tensor input,
    int64_t num_groups,
    double eps) {

    check_cuda_float_contiguous(input, "input");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t HW = H * W;

    TORCH_CHECK(C % num_groups == 0, "channels must be divisible by num_groups");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    group_norm_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, HW, num_groups,
        static_cast<float>(eps),
        stream);

    return output;
}

torch::Tensor histc_forward(
    torch::Tensor input,
    int64_t bins,
    double min,
    double max) {

    check_cuda_float_contiguous(input, "input");
    TORCH_CHECK(bins > 0, "bins must be positive");

    auto output = torch::empty({bins}, input.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    histc_launcher(
        input.data_ptr<float>(),
        input.numel(),
        static_cast<float>(min),
        static_cast<float>(max),
        bins,
        output.data_ptr<float>(),
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm_forward", &batch_norm_forward,
          "BatchNorm2d inference forward",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps"));
    m.def("group_norm_forward", &group_norm_forward,
          "GroupNorm forward without affine",
          py::arg("input"),
          py::arg("num_groups"),
          py::arg("eps"));
    m.def("histc_forward", &histc_forward,
          "Histogram forward",
          py::arg("input"),
          py::arg("bins"),
          py::arg("min"),
          py::arg("max"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
