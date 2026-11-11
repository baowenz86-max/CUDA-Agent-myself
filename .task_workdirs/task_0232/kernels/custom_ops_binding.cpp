#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void instance_norm_forward_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    int N, int C, int HW, float eps,
    cudaStream_t stream);

extern "C" void mul_scalar_launcher(
    const float* x,
    float* y,
    float multiplier,
    size_t numel,
    cudaStream_t stream);

// Wrapper for instance normalization
static torch::Tensor instance_norm_forward(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps) {
    // Check device
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    // Check dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    // Check contiguity
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    // Check dimensions
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1), "weight must be 1D of size C");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == x.size(1), "bias must be 1D of size C");

    auto N = static_cast<int>(x.size(0));
    auto C = static_cast<int>(x.size(1));
    auto HW = static_cast<int>(x.size(2) * x.size(3));

    auto y = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    instance_norm_forward_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, HW, static_cast<float>(eps),
        stream);

    return y;
}

// Wrapper for scalar multiplication
static torch::Tensor mul_scalar(
    const torch::Tensor& x,
    double multiplier) {
    // Check device
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    // Check dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    // Check contiguity
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto y = torch::empty_like(x);
    size_t numel = static_cast<size_t>(x.numel());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mul_scalar_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<float>(multiplier),
        numel,
        stream);

    return y;
}

// Register bindings
static void register_custom_ops(pybind11::module& m) {
    m.def("instance_norm_forward", &instance_norm_forward,
          "Instance normalization forward",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("eps"));
    m.def("mul_scalar", &mul_scalar,
          "Multiply tensor by scalar",
          pybind11::arg("x"),
          pybind11::arg("multiplier"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
