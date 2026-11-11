#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void instance_norm_forward_launcher(
    float* out,
    const float* x,
    const float* weight,
    const float* bias,
    float eps,
    int N, int C, int L,
    long long x_stride_n, long long x_stride_c, long long x_stride_l,
    long long out_stride_n, long long out_stride_c, long long out_stride_l,
    cudaStream_t stream);

torch::Tensor instance_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps = 1e-5) {
    // Validate inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N, C, L)");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int N = x.size(0);
    int C = x.size(1);
    int L = x.size(2);
    TORCH_CHECK(weight.size(0) == C, "weight size mismatch");
    TORCH_CHECK(bias.size(0) == C, "bias size mismatch");

    // Allocate output with same shape as x, contiguous
    auto out = torch::empty_like(x);
    TORCH_CHECK(out.is_contiguous(), "Output allocation should be contiguous");

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher, pass raw pointers
    instance_norm_forward_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        static_cast<float>(eps),
        N, C, L,
        x.stride(0), x.stride(1), x.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        stream
    );

    return out;
}

void register_instance_norm(pybind11::module& m) {
    m.def("instance_norm_forward", &instance_norm_forward,
          "InstanceNorm1d forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("eps") = 1e-05);
}

REGISTER_BINDING(instance_norm, register_instance_norm);
