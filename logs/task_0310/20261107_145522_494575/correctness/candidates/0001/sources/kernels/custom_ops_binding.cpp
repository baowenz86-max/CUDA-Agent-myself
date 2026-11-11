#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void instance_norm_forward_cuda_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int N, int C, int H, int W,
    long long sN, long long sC, long long sH, long long sW,
    float eps,
    cudaStream_t stream);

extern "C" void selu_forward_cuda_launcher(
    const float* x,
    float* y,
    long long numel,
    cudaStream_t stream);

torch::Tensor instance_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    // Ensure 4D input [N, C, H, W]
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    TORCH_CHECK(N > 0 && C > 0 && H > 0 && W > 0, "dimensions must be positive");
    TORCH_CHECK(weight.numel() == C, "weight must have C elements");
    TORCH_CHECK(bias.numel() == C, "bias must have C elements");

    // Allocate output tensor with same shape as x
    auto out = torch::empty_like(x);

    // Get strides for x (contiguous, but still compute)
    long long sN = x.stride(0);
    long long sC = x.stride(1);
    long long sH = x.stride(2);
    long long sW = x.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    instance_norm_forward_cuda_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        sN, sC, sH, sW,
        static_cast<float>(eps),
        stream
    );

    return out;
}

torch::Tensor selu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto out = torch::empty_like(x);
    long long numel = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    selu_forward_cuda_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        numel,
        stream
    );

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("instance_norm_forward", &instance_norm_forward,
          "InstanceNorm forward",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("eps"));
    m.def("selu_forward", &selu_forward,
          "SELU forward",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
