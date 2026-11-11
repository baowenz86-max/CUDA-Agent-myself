#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_group_norm_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C, int H, int W,
    float eps,
    cudaStream_t stream);

extern "C" void custom_tanh_launcher(
    float* output,
    const float* input,
    size_t numel,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// custom_group_norm
// -----------------------------------------------------------------------------
torch::Tensor custom_group_norm(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int num_groups,
    float eps) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(C % num_groups == 0, "num_groups must divide channels");

    // Weight and bias are optional (nullable). Validate if provided.
    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    if (weight.defined()) {
        TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
        TORCH_CHECK(weight.numel() == C, "weight must have length C");
        weight_ptr = weight.data_ptr<float>();
    }
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.numel() == C, "bias must have length C");
        bias_ptr = bias.data_ptr<float>();
    }

    // Allocate output tensor
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher (int32 for N,C,H,W is sufficient; launcher expects int)
    custom_group_norm_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<float>(eps),
        stream);

    return output;
}

// -----------------------------------------------------------------------------
// custom_tanh
// -----------------------------------------------------------------------------
torch::Tensor custom_tanh(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_tanh_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<size_t>(x.numel()),
        stream);

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("custom_group_norm", &custom_group_norm,
          "Custom GroupNorm forward",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("num_groups"), py::arg("eps"));
    m.def("custom_tanh", &custom_tanh,
          "Custom tanh forward",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
