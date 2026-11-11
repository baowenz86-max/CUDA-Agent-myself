// kernels/custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void replace_inf_with_nan_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream);

extern "C" void group_norm_forward_launcher(
    float* output,
    const float* x,
    const float* weight,
    const float* bias,
    int64_t N,
    int64_t C,
    int64_t HW,
    float eps,
    cudaStream_t stream);

// Wrapper for replace_inf_with_nan
torch::Tensor replace_inf_with_nan(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);

    int64_t numel = x.numel();
    if (numel == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    replace_inf_with_nan_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "replace_inf_with_nan kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for group_norm_forward
torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t HW = H * W;

    TORCH_CHECK(C % num_groups == 0, "num_groups must divide C");
    TORCH_CHECK(weight.numel() == C, "weight length must equal C");
    TORCH_CHECK(bias.numel() == C, "bias length must equal C");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    group_norm_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N,
        C,
        HW,
        static_cast<float>(eps),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "group_norm_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("replace_inf_with_nan", &replace_inf_with_nan,
          "Replace inf with NaN",
          py::arg("x"));
    m.def("group_norm_forward", &group_norm_forward,
          "GroupNorm forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("num_groups"),
          py::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
