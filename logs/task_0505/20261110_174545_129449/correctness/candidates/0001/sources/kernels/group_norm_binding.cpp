#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void group_norm_forward_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    int64_t N,
    int64_t C,
    int64_t HW,
    int64_t num_groups,
    double eps,
    cudaStream_t stream);

// PyTorch binding function
torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");

    // Ensure tensors are contiguous (the kernel assumes contiguous memory layout)
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    // Dimensions
    int64_t N = x_contig.size(0);
    int64_t C = x_contig.size(1);
    int64_t total = x_contig.numel();
    int64_t HW = total / (N * C);

    TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");

    // Allocate output with same shape as input (contiguous)
    auto y = torch::empty_like(x_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    group_norm_forward_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        N,
        C,
        HW,
        num_groups,
        eps,
        stream);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return y;
}

// Registration
void register_group_norm(pybind11::module& m) {
    m.def("group_norm_forward", &group_norm_forward,
          "Group Normalization forward (CUDA)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("num_groups"),
          py::arg("eps"));
}

REGISTER_BINDING(group_norm, register_group_norm);
