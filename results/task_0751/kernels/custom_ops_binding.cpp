#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    cudaStream_t stream);

extern "C" void minimum_forward_launcher(
    const float* input,
    const float* other,
    float* output,
    int64_t numel,
    int64_t out_features,
    cudaStream_t stream);

namespace {

// Helper to validate CUDA float32 contiguous tensor
torch::Tensor check_cuda_float32_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    return t;
}

} // anonymous namespace

// linear_forward: y = input @ weight^T + bias
torch::Tensor linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t in_features,
    int64_t out_features) {
    check_cuda_float32_contiguous(input, "input");
    check_cuda_float32_contiguous(weight, "weight");
    check_cuda_float32_contiguous(bias, "bias");

    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch_size, in_features]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [out_features, in_features]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [out_features]");

    int64_t batch_size = input.size(0);
    TORCH_CHECK(input.size(1) == in_features, "input.size(1) must equal in_features");
    TORCH_CHECK(weight.size(0) == out_features, "weight.size(0) must equal out_features");
    TORCH_CHECK(weight.size(1) == in_features, "weight.size(1) must equal in_features");
    TORCH_CHECK(bias.size(0) == out_features, "bias.size(0) must equal out_features");

    auto output = torch::empty({batch_size, out_features}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    linear_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        stream);

    // Surface launch errors immediately
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "linear_forward launcher failed: ", cudaGetErrorString(err));

    return output;
}

// minimum_forward: elementwise minimum with broadcasting along last dim
torch::Tensor minimum_forward(torch::Tensor input, torch::Tensor other) {
    check_cuda_float32_contiguous(input, "input");
    check_cuda_float32_contiguous(other, "other");

    TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
    TORCH_CHECK(other.dim() == 1, "other must be 1D");
    TORCH_CHECK(input.size(-1) == other.size(0), "input.size(-1) must equal other.size(0)");

    auto output = torch::empty_like(input);
    int64_t numel = input.numel();
    int64_t out_features = other.size(0);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    minimum_forward_launcher(
        input.data_ptr<float>(),
        other.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        out_features,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "minimum_forward launcher failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("linear_forward", &linear_forward,
          "Linear forward: y = input @ weight^T + bias",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("in_features"), py::arg("out_features"));
    m.def("minimum_forward", &minimum_forward,
          "Elementwise minimum with broadcasting along last dim",
          py::arg("input"), py::arg("other"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);