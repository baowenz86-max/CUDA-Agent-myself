#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration matches kernels/custom_ops.cu
extern "C" void batchnorm1d_inference_launcher(
    float* y,
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t L,
    cudaStream_t stream);

torch::Tensor batchnorm1d_inference_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");

    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");

    TORCH_CHECK(x.dim() == 3, "x must be 3D (N, C, L)");
    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t L = x.size(2);

    TORCH_CHECK(weight.numel() == C, "weight must have C elements");
    TORCH_CHECK(bias.numel() == C, "bias must have C elements");
    TORCH_CHECK(running_mean.numel() == C, "running_mean must have C elements");
    TORCH_CHECK(running_var.numel() == C, "running_var must have C elements");

    // Allocate output
    auto y = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    batchnorm1d_inference_launcher(
        y.data_ptr<float>(),
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        N,
        C,
        L,
        stream);

    // Check for launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");

    return y;
}

void register_batchnorm1d_inference(pybind11::module& m) {
    m.def("batchnorm1d_inference_forward", &batchnorm1d_inference_forward,
          "BatchNorm1d inference forward",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("running_mean"),
          pybind11::arg("running_var"),
          pybind11::arg("eps"));
}

REGISTER_BINDING(batchnorm1d_inference, register_batchnorm1d_inference);
