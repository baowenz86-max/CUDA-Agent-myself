// PyTorch binding for GroupNorm CUDA implementation
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of CUDA launcher from custom_ops.cu
extern "C" void group_norm_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t num_groups,
    double eps,
    cudaStream_t stream
);

// PyTorch wrapper function
torch::Tensor group_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps
) {
    // Validate input tensor
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4-dimensional (N,C,H,W)");

    // Validate weight and bias
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(weight.dim() == 1, "Weight must be 1-dimensional");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1-dimensional");

    int64_t C = input.size(1);
    TORCH_CHECK(weight.size(0) == C, "Weight size must match input channels");
    TORCH_CHECK(bias.size(0) == C, "Bias size must match input channels");
    TORCH_CHECK(C % num_groups == 0, "num_groups must divide channels");

    // Ensure input is contiguous for the kernel (kernel assumes contiguous but we can handle strides internally?)
    // The kernel uses linear indexing, so we need contiguous input.
    torch::Tensor input_contig = input.contiguous();
    torch::Tensor weight_contig = weight.contiguous();
    torch::Tensor bias_contig = bias.contiguous();

    // Allocate output tensor with same shape and device as input (contiguous)
    auto output = torch::empty_like(input_contig);

    int64_t N = input_contig.size(0);
    int64_t H = input_contig.size(2);
    int64_t W = input_contig.size(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    group_norm_forward_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, num_groups, eps, stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "GroupNorm CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_group_norm(pybind11::module& m) {
    m.def("group_norm_forward", &group_norm_forward,
          "GroupNorm forward",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("num_groups"),
          py::arg("eps"));
}

// Auto-register
REGISTER_BINDING(group_norm, register_group_norm);
