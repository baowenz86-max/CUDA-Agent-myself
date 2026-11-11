// custom_ops_binding.cpp
// Binding for GroupNorm CUDA kernel implemented in custom_ops.cu
// This file contains PyTorch tensor handling and Python bindings only.
// No CUDA kernels are implemented here.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void group_norm_forward_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int N,
    int C,
    int H,
    int W,
    int num_groups,
    float eps,
    cudaStream_t stream
);

// PyTorch wrapper function
torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps
) {
    // Parameter checks
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(C == weight.size(0), "weight size must match C");
    TORCH_CHECK(C == bias.size(0), "bias size must match C");
    TORCH_CHECK(num_groups > 0 && C % num_groups == 0, "num_groups must divide C");

    // Allocate output tensor
    auto y = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    group_norm_forward_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(num_groups),
        static_cast<float>(eps),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after group_norm_forward_launcher: ", cudaGetErrorString(err));

    return y;
}

// Registration function
void register_group_norm_forward(pybind11::module& m) {
    m.def(
        "group_norm_forward",
        &group_norm_forward,
        "GroupNorm forward",
        py::arg("x"),
        py::arg("weight"),
        py::arg("bias"),
        py::arg("num_groups"),
        py::arg("eps")
    );
}

// Auto-register
REGISTER_BINDING(group_norm_forward, register_group_norm_forward);
