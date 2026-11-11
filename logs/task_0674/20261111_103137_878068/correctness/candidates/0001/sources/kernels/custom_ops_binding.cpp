#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of the CUDA launcher from custom_ops.cu
extern "C" void batchnorm2d_eval_launcher(
    float* output,
    const float* input,
    const float* weight,   // may be nullptr
    const float* bias,     // may be nullptr
    const float* running_mean,
    const float* running_var,
    int N, int C, int H, int W,
    float eps,
    cudaStream_t stream);

// PyTorch wrapper for eval-mode BatchNorm2d.
// Parameters mirror model_new.py call: input, weight, bias, running_mean, running_var, eps.
torch::Tensor batchnorm2d_eval(
    torch::Tensor input,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must have shape [N, C, H, W]");
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    TORCH_CHECK(N > 0 && C > 0 && H > 0 && W > 0, "input dimensions must be positive");

    // Validate running_mean and running_var
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_mean.numel() == C, "running_mean must have exactly C elements");

    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");
    TORCH_CHECK(running_var.numel() == C, "running_var must have exactly C elements");

    // Optional weight and bias
    const float* weight_ptr = nullptr;
    if (weight.has_value()) {
        torch::Tensor w = weight.value();
        TORCH_CHECK(w.is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(w.dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(w.is_contiguous(), "weight must be contiguous");
        TORCH_CHECK(w.numel() == C, "weight must have exactly C elements");
        weight_ptr = w.data_ptr<float>();
    }

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        torch::Tensor b = bias.value();
        TORCH_CHECK(b.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(b.numel() == C, "bias must have exactly C elements");
        bias_ptr = b.data_ptr<float>();
    }

    // Allocate output with same shape, dtype, device, and strides as input
    auto output = torch::empty_like(input);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch the CUDA kernel via launcher
    batchnorm2d_eval_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<float>(eps),
        stream);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_batchnorm2d_eval(pybind11::module& m) {
    m.def(
        "batchnorm2d_eval",
        &batchnorm2d_eval,
        "Eval-mode BatchNorm2d: y = (x - running_mean) / sqrt(running_var + eps) * weight + bias",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("bias"),
        pybind11::arg("running_mean"),
        pybind11::arg("running_var"),
        pybind11::arg("eps") = 1e-05);
}

// Automatic registration via binding registry
REGISTER_BINDING(batchnorm2d_eval, register_batchnorm2d_eval);
