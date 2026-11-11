#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Declare the CUDA launcher from custom_ops.cu
extern "C" void group_norm_forward_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    int N, int C, int H, int W,
    int64_t stride_n_in, int64_t stride_c_in, int64_t stride_h_in, int64_t stride_w_in,
    int64_t stride_n_out, int64_t stride_c_out, int64_t stride_h_out, int64_t stride_w_out,
    int num_groups,
    double eps_double,
    cudaStream_t stream);

// PyTorch wrapper function
torch::Tensor group_norm_forward(
    torch::Tensor x,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    int64_t num_groups,
    double eps)
{
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    TORCH_CHECK(C % num_groups == 0, "num_groups must divide C");
    TORCH_CHECK(std::isfinite(eps), "eps must be finite");

    // Validate optional weight/bias
    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;

    if (weight.has_value()) {
        const torch::Tensor& w = weight.value();
        TORCH_CHECK(w.is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(w.dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(w.is_contiguous(), "weight must be contiguous");
        TORCH_CHECK(w.numel() == C, "weight must have exactly C elements");
        weight_ptr = w.data_ptr<float>();
    }

    if (bias.has_value()) {
        const torch::Tensor& b = bias.value();
        TORCH_CHECK(b.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(b.numel() == C, "bias must have exactly C elements");
        bias_ptr = b.data_ptr<float>();
    }

    // Allocate output with same shape (torch::empty_like preserves shape but may be contiguous)
    auto y = torch::empty_like(x);

    // Get input strides (note: x may be non-contiguous, launcher uses these strides)
    int64_t stride_n_in = x.stride(0);
    int64_t stride_c_in = x.stride(1);
    int64_t stride_h_in = x.stride(2);
    int64_t stride_w_in = x.stride(3);

    // Output strides (empty_like creates contiguous tensor, but we pass actual strides)
    int64_t stride_n_out = y.stride(0);
    int64_t stride_c_out = y.stride(1);
    int64_t stride_h_out = y.stride(2);
    int64_t stride_w_out = y.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    group_norm_forward_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        N, C, H, W,
        stride_n_in, stride_c_in, stride_h_in, stride_w_in,
        stride_n_out, stride_c_out, stride_h_out, stride_w_out,
        static_cast<int>(num_groups),
        eps,
        stream);

    // Check for kernel launch errors (optional but recommended)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return y;
}

// Registration function
void register_group_norm_forward(pybind11::module& m) {
    m.def("group_norm_forward", &group_norm_forward,
          "GroupNorm forward (num_groups=3, num_channels=3, epsilon=1e-5)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("num_groups"),
          py::arg("eps"));
}

// Auto-register with the infrastructure
REGISTER_BINDING(group_norm_forward, register_group_norm_forward);
