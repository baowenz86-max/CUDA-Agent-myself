// kernels/custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher from custom_ops.cu
extern "C" void batch_norm_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream
);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> batch_norm(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training = false,
    double momentum = 0.1,
    double eps = 1e-5
) {
    // Validate device and dtype
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");

    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");

    // Input must be 4D (N, C, H, W)
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    // Parameter tensors must have shape [C]
    TORCH_CHECK(weight.size(0) == C, "weight size mismatch");
    TORCH_CHECK(bias.size(0) == C, "bias size mismatch");
    TORCH_CHECK(running_mean.size(0) == C, "running_mean size mismatch");
    TORCH_CHECK(running_var.size(0) == C, "running_var size mismatch");

    // Parameters must be contiguous (likely already)
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");

    // Input can be non-contiguous? The reference input is contiguous, but launcher expects contiguous.
    // For safety, make contiguous if not.
    input = input.contiguous();

    // Allocate output tensor (same shape as input)
    auto output = torch::empty_like(input);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    batch_norm_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        N,
        C,
        H,
        W,
        stream
    );

    // Check for CUDA errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    // Return (output, running_mean, running_var) - note: training is typically false, so return original running stats
    return std::make_tuple(output, running_mean, running_var);
}

void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm", &batch_norm,
          "Batch normalization forward (inference mode)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("training") = false,
          py::arg("momentum") = 0.1,
          py::arg("eps") = 1e-5);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
