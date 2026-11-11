#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void group_norm_forward_launcher(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    int N, int C, int H, int W,
    int num_groups,
    float eps,
    cudaStream_t stream
);

extern "C" void gelu_forward_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream
);

// GroupNorm forward binding
torch::Tensor group_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Validate group parameters
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    TORCH_CHECK(C % num_groups == 0, "num_groups must divide C");

    // Validate weight and bias
    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    if (weight.defined()) {
        TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
        TORCH_CHECK(weight.numel() == C, "weight must have C elements");
        weight_ptr = weight.data_ptr<float>();
    }
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.numel() == C, "bias must have C elements");
        bias_ptr = bias.data_ptr<float>();
    }

    // Allocate output with same shape and options
    auto output = torch::empty_like(input);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    group_norm_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        N, C, H, W,
        static_cast<int>(num_groups),
        static_cast<float>(eps),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "group_norm_forward launcher failed: ", cudaGetErrorString(err));

    return output;
}

// GELU forward binding
torch::Tensor gelu_forward(torch::Tensor input) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    // Allocate output
    auto output = torch::empty_like(input);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    int total = static_cast<int>(input.numel());
    gelu_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gelu_forward launcher failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("group_norm_forward", &group_norm_forward,
          "GroupNorm forward",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("num_groups"),
          pybind11::arg("eps"));
    m.def("gelu_forward", &gelu_forward,
          "GELU forward",
          pybind11::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
