#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declared in custom_ops.cu
extern "C" void group_norm_forward_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int N,
    int C,
    int H,
    int W,
    double eps,
    cudaStream_t stream);

torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps) {
    // Validate inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1), "weight must be 1D of length C");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == x.size(1), "bias must be 1D of length C");
    // Launcher expects N,C,H,W order
    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);

    // Allocate output tensor with same shape and options as x
    auto y = torch::empty_like(x);

    // Ensure tensors are contiguous (launcher assumes contiguous layout)
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    group_norm_forward_launcher(
        x_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        N,
        C,
        H,
        W,
        eps,
        stream);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return y;
}

void register_group_norm(pybind11::module& m) {
    m.def("group_norm_forward", &group_norm_forward,
          "GroupNorm forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps"));
}

REGISTER_BINDING(group_norm, register_group_norm);
